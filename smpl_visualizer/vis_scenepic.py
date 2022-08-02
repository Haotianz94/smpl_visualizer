import torch
import numpy as np
import scenepic as sp
from .smpl import SMPL, SMPL_MODEL_DIR, BASE_DIR
from .torch_transform import quat_apply, quat_between_two_vec, quaternion_to_angle_axis, angle_axis_to_quaternion
from vtk import vtkTransform
from .vis import make_checker_board_texture, get_color_palette
from tqdm import tqdm
from PIL import Image

NET_TEXTURE_PATH = BASE_DIR + 'net_texture.png'

class SportVisualizerHTML():
    def __init__(self, device=torch.device('cpu')):
        self.smpl_female = SMPL(SMPL_MODEL_DIR, gender='female', create_transl=True).to(device)
        self.smpl_faces = self.smpl_female.faces
        self.smpl = SMPL(SMPL_MODEL_DIR, pose_type='body26fk', create_transl=False, gender='f').to(device)
        self.racket_params = None
    
    def get_camera_intrinsics(self, width=1920, height=1080):
        fx = fy = max(width, height)
        K = np.diag([fx, fy, 1.0])
        K[0, 2] = width / 2
        K[1, 2] = height / 2
        return K

    def load_default_camera(self, intrinsics):
        # this function loads an "OpenCV"-style camera representation
        # and converts it to a GL style for use in ScenePic
        # location = np.array(camera_info["location"], np.float32)
        # euler_angles = np.array(camera_info["rotation"], np.float32)
        # rotation = sp.Transforms.euler_angles_to_matrix(euler_angles, "XYZ")
        # translation = sp.Transforms.translate(location)
        # extrinsics = translation @ rotation

        img_width = intrinsics[0,2] * 2
        img_height = intrinsics[1,2] * 2

        aspect_ratio = img_width / img_height
       
        return sp.Camera(center=(10, 0, 3), look_at=(0, 0, 0), up_dir=(0, 0, 1), 
                         fov_y_degrees=45.0, aspect_ratio=aspect_ratio, far_crop_distance=40)

    def load_camera_from_ext_int(self, extrinsics, intrinsics):
        # this function loads an "OpenCV"-style camera representation
        # and converts it to a GL style for use in ScenePic
        # location = np.array(camera_info["location"], np.float32)
        # euler_angles = np.array(camera_info["rotation"], np.float32)
        # rotation = sp.Transforms.euler_angles_to_matrix(euler_angles, "XYZ")
        # translation = sp.Transforms.translate(location)
        # extrinsics = translation @ rotation

        img_width = intrinsics[0,2] * 2
        img_height = intrinsics[1,2] * 2
        fx = intrinsics[0,0]
        fy = intrinsics[1,1]

        # pred_f_pix = orig_img_h / 2. / np.tan(pred_vfov / 2.)
        
        vfov = 2 * np.arctan(2 * fx / img_height)
        world_to_camera = sp.Transforms.gl_world_to_camera(extrinsics)
        aspect_ratio = img_width / img_height
        projection = sp.Transforms.gl_projection(45, aspect_ratio, 0.01, 100)

        return sp.Camera(world_to_camera, projection)

    def init_players_and_rackets(self, smpl_seq=None, racket_seq=None):
        self.smpl_seq = smpl_seq
        self.smpl_verts = None

        if 'joint_rot' in smpl_seq:
            joint_rot = smpl_seq['joint_rot'] # num_actor x num_frames x (num_joints x 3)
            trans = smpl_seq['trans'] # num_actor x num_frames x 3
            
            self.smpl_motion = self.smpl(
                global_orient=joint_rot[..., :3].view(-1, 3),
                body_pose=joint_rot[..., 3:].view(-1, 69),
                betas=torch.zeros(joint_rot.shape[0]*joint_rot.shape[1], 10).float(),
                root_trans = trans.view(-1, 3),
                return_full_pose=True,
                orig_joints=True
            )

            self.smpl_verts = self.smpl_motion.vertices.reshape(*joint_rot.shape[:-1], -1, 3)
            self.smpl_joints = self.smpl_motion.joints.reshape(*joint_rot.shape[:-1], -1, 3)

        if racket_seq is not None:
            num_actors, num_frames = trans.shape[:2]
            for i in range(num_actors):
                for j in range(num_frames):
                    if racket_seq[i][j] is not None:
                        racket_seq[i][j]['root'] = trans[i, j].numpy()
            self.racket_params = racket_seq

        if self.correct_root_height:
            diff_root_height = torch.min(self.smpl_joints[:, :, 10:12, 2], dim=2)[0].view(*trans.shape[:2], 1)
            self.smpl_joints[:, :, :, 2] -= diff_root_height
            if self.smpl_verts is not None:
                self.smpl_verts[:, :, :, 2] -= diff_root_height

        self.num_actors = self.smpl_joints.shape[0]
        self.num_fr = self.smpl_joints.shape[1]
    
    def create_court(self, scene, frame):
        if self.sport == 'tennis':
            # Court
            wlh = (10.97, 11.89*2, 0.05)
            center = np.array([0, 0, -wlh[2] * 0.5])
            court_mesh = pyvista.Cube(center, *wlh)
            self.pl.add_mesh(court_mesh, color='#4A609D', ambient=0.2, diffuse=0.8, specular=0, smooth_shading=True)

            # Court lines (vertical)
            for x, l in zip([-10.97/2, -8.23/2, 0, 8.23/2, 10.97/2], [23.77, 23.77, 12.8, 23.77, 23.77]):
                wlh = (0.05, l, 0.05)
                center = np.array([x, 0, -wlh[2] * 0.5])
                court_line_mesh = pyvista.Cube(center, *wlh)
                court_line_mesh.points[:, 2] += 0.01
                self.pl.add_mesh(court_line_mesh, color='#FFFFFF', ambient=0.2, diffuse=0.8, specular=0.8, specular_power=5, smooth_shading=True)
            
            # Court lines (horizontal)
            for y, w in zip([-11.89, -6.4, 0, 6.4, 11.89], [10.97, 8.23, 10.97, 8.23, 10.97]):
                wlh = (w, 0.05, 0.05)
                center = np.array([0, y, -wlh[2] * 0.5])
                court_line_mesh = pyvista.Cube(center, *wlh)
                court_line_mesh.points[:, 2] += 0.01
                self.pl.add_mesh(court_line_mesh, color='#FFFFFF', ambient=0.2, diffuse=0.8, specular=0.8, specular_power=5, smooth_shading=True)
            
            # Post
            for x in [-0.91-10.97/2, 0.91+10.97/2]:
                wlh = (0.05, 0.05, 1.2)
                center = np.array([x, 0, wlh[2] * 0.5])
                post_mesh = pyvista.Cube(center, *wlh)
                self.pl.add_mesh(post_mesh, color='#BD7427', ambient=0.2, diffuse=0.8, specular=0.8, specular_power=5, smooth_shading=True)
            
            # Net
            wlh = (10.97+0.91*2, 0.01, 1.07)
            center = np.array([0, 0, 1.07/2])
            net_mesh = pyvista.Cube(center, *wlh)
            net_mesh.active_t_coords *= 1000
            tex = pyvista.numpy_to_texture(make_checker_board_texture('#FFFFFF', '#AAAAAA', width=10, height=10))
            self.pl.add_mesh(net_mesh, texture=tex, ambient=0.2, diffuse=0.8, opacity=0.1, smooth_shading=True)

        elif self.sport == 'badminton':
            # Court
            court_mesh = scene.create_mesh()
            wlh = [6.1, 13.41, 0.05]
            center = np.array([0, 0, -wlh[2] * 0.5])
            court_mesh.add_cube(
                color=np.array([74, 96, 157]) / 255., 
                transform=np.dot(sp.Transforms.Translate(center), sp.Transforms.Scale(wlh)))

            # Court lines (vertical)
            for x in [-3.05, -2.6, 2.6, 3.05]:
                wlh = (0.05, 13.41, 0.05)
                center = np.array([x, 0, -wlh[2] * 0.5 + 0.01])
                court_mesh.add_cube(
                    color=(1, 1, 1), 
                    transform=np.dot(sp.Transforms.Translate(center), sp.Transforms.Scale(wlh)))
            for x, y, l in zip([0, 0], [-(1.98+6.71)/2, (1.98+6.71)/2], [3.96+0.76, 3.96+0.76]):
                wlh = (0.05, l, 0.05)
                center = np.array([x, y, -wlh[2] * 0.5 + 0.01])
                court_mesh.add_cube(
                    color=(1, 1, 1), 
                    transform=np.dot(sp.Transforms.Translate(center), sp.Transforms.Scale(wlh)))
            
            # Court lines (horizontal)
            for y in [-6.71, -3.96-1.98, -1.98, 1.98, 1.98+3.96, 6.71]:
                wlh = (6.1, 0.05, 0.05)
                center = np.array([0, y, -wlh[2] * 0.5 + 0.01])
                court_mesh.add_cube(
                    color=(1, 1, 1), 
                    transform=np.dot(sp.Transforms.Translate(center), sp.Transforms.Scale(wlh)))

            frame.add_mesh(court_mesh)

            # Post
            post_mesh = scene.create_mesh()
            for x in [-3.05, 3.05]:
                wlh = (0.05, 0.05, 1.55)
                center = np.array([x, 0, wlh[2] * 0.5])
                post_mesh.add_cube(
                    color=np.array([189, 116, 39]) / 255., 
                    transform=np.dot(sp.Transforms.Translate(center), sp.Transforms.Scale(wlh)))
            frame.add_mesh(post_mesh)

            # Net
            wlh = (6.1, 0.01, 0.79)
            center = np.array([0, 0, (0.76+1.55)/2])

            net_texture = scene.create_image(image_id="net")
            # checker_board = make_checker_board_texture('#FFFFFF', '#AAAAAA', width=10)
            # checker_board = make_checker_board_texture('#FF0000', '#00FF00', width=1000, alpha=0.5)
            # img = Image.fromarray(checker_board, 'RGBA') 
            # img.save('net.png')
            # net_texture.from_numpy(checker_board)
            net_texture.load(NET_TEXTURE_PATH)
            net_mesh = scene.create_mesh(double_sided=True, use_texture_alpha=True, texture_id='net')
            net_mesh.add_image(
                origin=(-6.1/2, 0, 0.76),
                x_axis=(6.1, 0, 0),
                y_axis=(0, 0, 0.79),
                uv_1=(1, 0),
                uv_2=(1, 0.79/6.1),
                uv_3=(0, 0.79/6.1),
            )

            frame.add_mesh(net_mesh)

        # Floor
        floor_mesh = scene.create_mesh()
        wlh = (20, 40, 0.05)
        center = np.array([0, 0, -wlh[2] * 0.5 - 0.01])
        floor_mesh.add_cube(
            color=np.array([118, 151, 113]) / 255., 
            transform=np.dot(sp.Transforms.Translate([center]), sp.Transforms.Scale(wlh)))
        frame.add_mesh(floor_mesh)
    
    def create_racket(self, scene, params):
        if params is None: return None

        def get_transform(scale, trans, new_dir, old_dir=[1., 0., 0.]):
            trans = sp.Transforms.Translate(trans)
            scale = sp.Transforms.Scale(scale)
            new_dir = torch.from_numpy(new_dir).float()
            aa = quaternion_to_angle_axis(quat_between_two_vec(torch.tensor(old_dir).expand_as(new_dir), new_dir)).numpy()
            angle = np.linalg.norm(aa, axis=-1, keepdims=True)
            axis = aa / (angle + 1e-6)
            rotation = sp.Transforms.rotation_matrix_from_axis_angle(axis, angle)
            return trans.dot(rotation.dot(scale))

        if self.sport == 'tennis':
            pass
        elif self.sport == 'badminton':
            # self.head_actor.SetUserTransform(get_transform(params['head_center'] + params['root'], params['racket_normal']))
            # self.net_actor.SetUserTransform(get_transform(params['head_center'] + params['root'], params['racket_normal']))
            # self.shaft_actor.SetUserTransform(get_transform(params['shaft_center'] + params['root'], params['racket_dir']))
            # self.handle_actor.SetUserTransform(get_transform(params['handle_center'] + params['root'], params['racket_dir']))

            racket_mesh = scene.create_mesh()
            # Handle
            racket_mesh.add_cylinder(
                color=np.array([0, 0, 0]) / 255., 
                transform=get_transform(
                    scale=[0.15, 0.0254, 0.0254],
                    trans=params['handle_center'] + params['root'], 
                    new_dir=params['racket_dir']
            ))
            # Shaft
            racket_mesh.add_cylinder(
                color=np.array([0, 0, 0]) / 255., 
                transform=get_transform(
                    scale=[0.25, 0.01, 0.01],
                    trans=params['shaft_center'] + params['root'], 
                    new_dir=params['racket_dir']
            ))
            racket_mesh.add_disc(
                color=np.array([150, 150, 150]) / 255.,
                segment_count=50,
                fill_triangles=False,  
                add_wireframe=True,  
                transform=get_transform(
                    scale=[1, 0.28, 0.25],
                    trans=params['head_center'] + params['root'], 
                    new_dir=params['racket_normal']
            ))
            
        return racket_mesh

            
    def create_canvas(self, scene):
        canvas = scene.create_canvas_3d(width=self.image_width, height=self.image_height)
        cam_intrinsics = self.get_camera_intrinsics(self.image_width, self.image_height)
        
        for i in tqdm(range(self.num_fr)):
            frame = canvas.create_frame()

            # Add coordinate axis
            # coord_ax = scene.create_mesh()
            # coord_ax.add_coordinate_axes()
            # frame.add_mesh(coord_ax)
            
            self.create_court(scene, frame)

            colors = get_color_palette(self.num_actors, 'rainbow', use_float=True)
            for j in range(self.num_actors):
                if self.num_actors == 1:
                    smpl_mesh = scene.create_mesh(shared_color=(0.7, 0.7, 0.7))
                elif self.num_actors == 2:
                    smpl_mesh = scene.create_mesh(shared_color=(0.7, 0.7, 0.7) if j == 0 else (0.5, 0.5, 0.5))
                else:
                    smpl_mesh = scene.create_mesh(shared_color=colors[j])
                if self.smpl_seq['joint_rot'][j, i].sum() != 0:
                    smpl_mesh.add_mesh_without_normals(self.smpl_verts[j, i].contiguous().numpy(), self.smpl_faces)
                    frame.add_mesh(smpl_mesh)

                if self.racket_params is not None:
                    racket_mesh = self.create_racket(scene, self.racket_params[j][i])
                if racket_mesh is not None: frame.add_mesh(racket_mesh)

            frame.camera = self.load_default_camera(cam_intrinsics)

        return canvas

    def save_animation_as_html(self, init_args, html_path="demo.html"):
        scene = sp.Scene()

        self.image_width = init_args.get('image_width', 1920)
        self.image_height = init_args.get('image_height', 1080)
        self.sport = init_args.get('sport', 'tennis')
        self.correct_root_height = init_args.get('correct_root_height')
        self.init_players_and_rackets(smpl_seq=init_args.get('smpl_seq'), 
            racket_seq=init_args.get('racket_seq'))

        self.create_canvas(scene)
        # if smpl_seq_gt is not None:
        #     canvas_gt = self.create_canvas(scene, smpl_seq_gt, img_width, img_height, cam_intrinsics)
        # if smpl_seq_gt is not None:
        #     scene.link_canvas_events(canvas_pred, canvas_gt)
        scene.save_as_html(html_path, title="sport visualizer")
        print(f"Saved animation as html into {html_path}")