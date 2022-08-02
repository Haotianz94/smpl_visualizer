import pyvista
import torch
import numpy as np
from pyvista.plotting.tools import parse_color
from vtk import vtkTransform
from .torch_transform import quat_apply, quat_between_two_vec, quaternion_to_angle_axis, angle_axis_to_quaternion
from .vis_pyvista import PyvistaVisualizer
from .smpl import SMPL, SMPL_MODEL_DIR
from .vis import make_checker_board_texture, get_color_palette


class SMPLActor():

    def __init__(self, pl, verts, faces, color='#FF8A82', visible=True):
        self.pl = pl
        self.verts = verts
        self.face = faces
        self.mesh = pyvista.PolyData(verts, faces)
        self.actor = self.pl.add_mesh(self.mesh, color=color, pbr=True, metallic=0.0, roughness=0.3, diffuse=1)
        # self.actor = self.pl.add_mesh(self.mesh, color=color, ambient=0.2, diffuse=0.8, specular=0.8, specular_power=5, smooth_shading=True)
        self.set_visibility(visible)

    def update_verts(self, new_verts):
        self.mesh.points[...] = new_verts
        self.mesh.compute_normals(inplace=True)

    def set_opacity(self, opacity):
        self.actor.GetProperty().SetOpacity(opacity)

    def set_visibility(self, flag):
        self.actor.SetVisibility(flag)

    def set_color(self, color):
        rgb_color = parse_color(color)
        self.actor.GetProperty().SetColor(rgb_color)


class SkeletonActor():

    def __init__(self, pl, joint_parents, joint_color='green', bone_color='yellow', joint_radius=0.03, bone_radius=0.02, visible=True):
        self.pl = pl
        self.joint_parents = joint_parents
        self.joint_meshes = []
        self.joint_actors = []
        self.bone_meshes = []
        self.bone_actors = []
        self.bone_pairs = []
        for j, pa in enumerate(self.joint_parents):
            # joint
            joint_mesh = pyvista.Sphere(radius=joint_radius, center=(0, 0, 0), theta_resolution=10, phi_resolution=10)
            # joint_actor = self.pl.add_mesh(joint_mesh, color=joint_color, pbr=True, metallic=0.0, roughness=0.3, diffuse=1)
            joint_actor = self.pl.add_mesh(joint_mesh, color=joint_color, ambient=0.3, diffuse=0.5, specular=0.8, specular_power=5, smooth_shading=True)
            self.joint_meshes.append(joint_mesh)
            self.joint_actors.append(joint_actor)
            # bone
            if pa >= 0:
                bone_mesh = pyvista.Cylinder(radius=bone_radius, center=(0, 0, 0), direction=(0, 0, 1), resolution=30)
                # bone_actor = self.pl.add_mesh(bone_mesh, color=bone_color, pbr=True, metallic=0.0, roughness=0.3, diffuse=1)
                bone_actor = self.pl.add_mesh(bone_mesh, color=bone_color, ambient=0.3, diffuse=0.5, specular=0.8, specular_power=5, smooth_shading=True)
                self.bone_meshes.append(bone_mesh)
                self.bone_actors.append(bone_actor)
                self.bone_pairs.append((j, pa))
        self.set_visibility(visible)

    def update_joints(self, jpos):
        # joint
        for actor, pos in zip(self.joint_actors, jpos):
            trans = vtkTransform()
            trans.Translate(*pos)
            actor.SetUserTransform(trans)
        # bone
        vec = []
        for actor, (j, pa) in zip(self.bone_actors, self.bone_pairs):
            vec.append((jpos[j] - jpos[pa]))
        vec = np.stack(vec)
        dist = np.linalg.norm(vec, axis=-1)
        vec = torch.tensor(vec / dist[..., None])
        aa = quaternion_to_angle_axis(quat_between_two_vec(torch.tensor([0., 0., 1.]).expand_as(vec), vec)).numpy()
        angle = np.linalg.norm(aa, axis=-1, keepdims=True)
        axis = aa / (angle + 1e-6)
        
        for actor, (j, pa), angle_i, axis_i, dist_i in zip(self.bone_actors, self.bone_pairs, angle, axis, dist):
            trans = vtkTransform()
            trans.Translate(*(jpos[pa] + jpos[j]) * 0.5)
            trans.RotateWXYZ(np.rad2deg(angle_i), *axis_i)
            trans.Scale(1, 1, dist_i)
            actor.SetUserTransform(trans)

    def set_opacity(self, opacity):
        for actor in self.joint_actors:
            actor.GetProperty().SetOpacity(opacity)
        for actor in self.bone_actors:
            actor.GetProperty().SetOpacity(opacity)

    def set_visibility(self, flag):
        for actor in self.joint_actors:
            actor.SetVisibility(flag)
        for actor in self.bone_actors:
            actor.SetVisibility(flag)

    def set_color(self, color):
        rgb_color = parse_color(color)
        for actor in self.joint_actors:
            actor.GetProperty().SetColor(rgb_color)
        for actor in self.jbone_actors:
            actor.GetProperty().SetColor(rgb_color)


class RacketActor():

    def __init__(self, pl, sport='tennis'):
        self.pl = pl
        self.sport = sport
        if self.sport == 'badminton':
            self.net_mesh = pyvista.Cylinder(center=(0, 0, 0), radius=0.25/2, height=0.01, direction=(0, 0, 1))
            self.net_mesh.active_t_coords *= 1000
            tex = pyvista.numpy_to_texture(make_checker_board_texture('#FFFFFF', '#AAAAAA', width=10))
            self.net_actor = self.pl.add_mesh(self.net_mesh, texture=tex, ambient=0.2, diffuse=0.8, opacity=0.1, smooth_shading=True)

            self.head_mesh = pyvista.Tube(pointa=(0, 0, -0.005), pointb=(0, 0, 0.005), radius=0.25/2)
            self.head_actor = self.pl.add_mesh(self.head_mesh, color='black', ambient=0.3, diffuse=0.5, smooth_shading=True)

            self.shaft_mesh = pyvista.Cylinder(center=(0, 0, 0), radius=0.005, height=0.25, direction=(0, 0, 1))
            self.shaft_actor = self.pl.add_mesh(self.shaft_mesh, color='black', ambient=0.3, diffuse=0.5, smooth_shading=True)

            self.handle_mesh = pyvista.Cylinder(center=(0, 0, 0), radius=0.0254/2, height=0.15, direction=(0, 0, 1))
            self.handle_actor = self.pl.add_mesh(self.handle_mesh, color='#AAAAAA', ambient=0.3, diffuse=0.5, smooth_shading=True)
            
            self.actors = [self.head_actor, self.net_actor, self.shaft_actor, self.handle_actor]
        elif self.sport == 'tennis':
            self.net_mesh = pyvista.Cylinder(center=(0, 0, 0), radius=0.15, height=0.01, direction=(0, 0, 1))
            self.net_mesh.active_t_coords *= 1000
            tex = pyvista.numpy_to_texture(make_checker_board_texture('#FFFFFF', '#AAAAAA', width=10))
            self.net_actor = self.pl.add_mesh(self.net_mesh, texture=tex, ambient=0.2, diffuse=0.8, opacity=0.1, smooth_shading=True)

            self.head_mesh = pyvista.Tube(pointa=(0, 0, -0.01), pointb=(0, 0, 0.01), radius=0.15)
            self.head_actor = self.pl.add_mesh(self.head_mesh, color='black', ambient=0.3, diffuse=0.5, smooth_shading=True)

            self.shaft_left_mesh = pyvista.Cylinder(center=(0, 0, 0), radius=0.01, height=0.16/np.cos(np.pi/8), direction=(0, 0, 1))
            self.shaft_left_actor = self.pl.add_mesh(self.shaft_left_mesh, color='black', ambient=0.3, diffuse=0.5, smooth_shading=True)

            self.shaft_right_mesh = pyvista.Cylinder(center=(0, 0, 0), radius=0.01, height=0.16/np.cos(np.pi/8), direction=(0, 0, 1))
            self.shaft_right_actor = self.pl.add_mesh(self.shaft_right_mesh, color='black', ambient=0.3, diffuse=0.5, smooth_shading=True)

            self.handle_mesh = pyvista.Cylinder(center=(0, 0, 0), radius=0.02, height=0.16, direction=(0, 0, 1))
            self.handle_actor = self.pl.add_mesh(self.handle_mesh, color='black', ambient=0.3, diffuse=0.5, smooth_shading=True)
            
            self.actors = [self.head_actor, self.net_actor, self.shaft_left_actor, self.shaft_right_actor, self.handle_actor]


    def update_racket(self, params):
        def get_transform(new_pos, new_dir):
            trans = vtkTransform()
            trans.Translate(new_pos)
            new_dir = torch.from_numpy(new_dir).float()
            aa = quaternion_to_angle_axis(quat_between_two_vec(torch.tensor([0., 0., 1.]).expand_as(new_dir), new_dir)).numpy()
            angle = np.linalg.norm(aa, axis=-1, keepdims=True)
            axis = aa / (angle + 1e-6)
            trans.RotateWXYZ(np.rad2deg(angle), *axis)
            return trans

        if self.sport == 'badminton':
            self.head_actor.SetUserTransform(get_transform(params['head_center'] + params['root'], params['racket_normal']))
            self.net_actor.SetUserTransform(get_transform(params['head_center'] + params['root'], params['racket_normal']))
            self.shaft_actor.SetUserTransform(get_transform(params['shaft_center'] + params['root'], params['racket_dir']))
            self.handle_actor.SetUserTransform(get_transform(params['handle_center'] + params['root'], params['racket_dir']))
        elif self.sport == 'tennis':
            self.head_actor.SetUserTransform(get_transform(params['head_center'] + params['root'], params['racket_normal']))
            self.net_actor.SetUserTransform(get_transform(params['head_center'] + params['root'], params['racket_normal']))
            self.shaft_left_actor.SetUserTransform(get_transform(params['shaft_left_center'] + params['root'], params['shaft_left_dir']))
            self.shaft_right_actor.SetUserTransform(get_transform(params['shaft_right_center'] + params['root'], params['shaft_right_dir']))
            self.handle_actor.SetUserTransform(get_transform(params['handle_center'] + params['root'], params['racket_dir']))
    
    def set_visibility(self, flag):
        for actor in self.actors:
            actor.SetVisibility(flag)


class SportVisualizer(PyvistaVisualizer):

    def __init__(self, show_smpl=False, show_skeleton=True, show_racket=False, 
        correct_root_height=False, device=torch.device('cpu'), **kwargs):
        
        super().__init__(**kwargs)
        self.show_smpl = show_smpl
        self.show_skeleton = show_skeleton
        self.show_racket = show_racket
        self.correct_root_height = correct_root_height
        self.smpl = SMPL(SMPL_MODEL_DIR, pose_type='body26fk', create_transl=False, gender='f').to(device)
        faces = self.smpl.faces.copy()       
        self.smpl_faces = faces = np.hstack([np.ones_like(faces[:, [0]]) * 3, faces])
        self.smpl_joint_parents = self.smpl.parents.cpu().numpy()
        self.device = device
        
    def update_smpl_seq(self, smpl_seq=None, racket_seq=None):
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
            if 'joint_pos' not in smpl_seq:
                self.smpl_joints = self.smpl_motion.joints.reshape(*joint_rot.shape[:-1], -1, 3)

        if 'joint_pos' in smpl_seq:
            joints = smpl_seq['joint_pos'] # num_actor x num_frames x num_joints x 3
            trans = smpl_seq['trans'] # num_actor x num_frames x 3

            # Orient is None for hybrIK since joints already has global orentation             
            orient = smpl_seq['orient']

            joints_world = joints
            if orient is not None:
                joints_world = torch.cat([torch.zeros_like(joints[..., :3]), joints], dim=-1).view(*joints.shape[:-1], -1, 3)
                orient_q = angle_axis_to_quaternion(orient).unsqueeze(-2).expand(joints.shape[:-1] + (4,))
                joints_world = quat_apply(orient_q, joints_world)
            if trans is not None:
                joints_world = joints_world + trans.unsqueeze(-2)
            self.smpl_joints = joints_world
        
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

        self.fr = 0
        self.num_fr = self.smpl_joints.shape[1]

    def init_camera(self, init_args):
        super().init_camera()

        if init_args.get('sport') == 'tennis':
            if init_args.get('camera') == 'front':
                self.pl.camera.up = (0, 0, 1)
                self.pl.camera.focal_point = [0, 0, 0]
                self.pl.camera.position = [0, -25, 3]
            elif init_args.get('camera') == 'side_both':
                self.pl.camera.up = (0, 0, 1)
                self.pl.camera.focal_point = [0, 0, 0]
                self.pl.camera.position = [15, 0, 3]
            elif init_args.get('camera') == 'side_near':
                self.pl.camera.elevation = 0
                self.pl.camera.up = (0, 0, 1)
                self.pl.camera.focal_point = [0, -12, 0]
                # self.pl.camera.position = [15, -12, 3]
                self.pl.camera.position = [15, -12, 0]
            elif init_args.get('camera') == 'side_far':
                self.pl.camera.elevation = 0
                self.pl.camera.up = (0, 0, 1)
                self.pl.camera.focal_point = [0, 12, 0]
                # self.pl.camera.position = [15, 12, 3]
                self.pl.camera.position = [15, 12, 0]
        elif init_args.get('sport') == 'badminton':
            if init_args.get('camera') == 'front':
                self.pl.camera.up = (0, 0, 1)
                self.pl.camera.focal_point = [0, 0, 0]
                self.pl.camera.position = [0, -13, 3]
            elif init_args.get('camera') == 'side_both':
                self.pl.camera.up = (0, 0, 1)
                self.pl.camera.focal_point = [0, 0, 0]
                self.pl.camera.position = [15, 0, 3]
            elif init_args.get('camera') == 'side_near':
                self.pl.camera.elevation = 0
                self.pl.camera.up = (0, 0, 1)
                self.pl.camera.focal_point = [0, -3.5, 0]
                # self.pl.camera.position = [15, -3.5, 3]
                self.pl.camera.position = [15, -3.5, 0]
            elif init_args.get('camera') == 'side_far':
                self.pl.camera.elevation = 0
                self.pl.camera.up = (0, 0, 1)
                self.pl.camera.focal_point = [0, 3.5, 0]
                # self.pl.camera.position = [15, 3.5, 3]
                self.pl.camera.position = [15, 3.5, 0]

    def init_scene(self, init_args):
        if init_args is None:
            init_args = dict()
        super().init_scene(init_args)
        # Init tennis court
        if init_args.get('sport') == 'tennis':
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
            tex = pyvista.numpy_to_texture(make_checker_board_texture('#FFFFFF', '#AAAAAA', width=10))
            self.pl.add_mesh(net_mesh, texture=tex, ambient=0.2, diffuse=0.8, opacity=0.1, smooth_shading=True)

        elif init_args.get('sport') == 'badminton':
            # Court
            wlh = (6.1, 13.41, 0.05)
            center = np.array([0, 0, -wlh[2] * 0.5])
            court_mesh = pyvista.Cube(center, *wlh)
            self.pl.add_mesh(court_mesh, color='#4A609D', ambient=0.2, diffuse=0.8, specular=0, smooth_shading=True)

            # Court lines (vertical)
            for x in [-3.05, -2.6, 2.6, 3.05]:
                wlh = (0.05, 13.41, 0.05)
                center = np.array([x, 0, -wlh[2] * 0.5])
                court_line_mesh = pyvista.Cube(center, *wlh)
                court_line_mesh.points[:, 2] += 0.01
                self.pl.add_mesh(court_line_mesh, color='#FFFFFF', ambient=0.2, diffuse=0.8, specular=0.8, specular_power=5, smooth_shading=True)
            for x, y, l in zip([0, 0], [-(1.98+6.71)/2, (1.98+6.71)/2], [3.96+0.76, 3.96+0.76]):
                wlh = (0.05, l, 0.05)
                center = np.array([x, y, -wlh[2] * 0.5])
                court_line_mesh = pyvista.Cube(center, *wlh)
                court_line_mesh.points[:, 2] += 0.01
                self.pl.add_mesh(court_line_mesh, color='#FFFFFF', ambient=0.2, diffuse=0.8, specular=0.8, specular_power=5, smooth_shading=True)
            
            # Court lines (horizontal)
            for y in [-6.71, -3.96-1.98, -1.98, 1.98, 1.98+3.96, 6.71]:
                wlh = (6.1, 0.05, 0.05)
                center = np.array([0, y, -wlh[2] * 0.5])
                court_line_mesh = pyvista.Cube(center, *wlh)
                court_line_mesh.points[:, 2] += 0.01
                self.pl.add_mesh(court_line_mesh, color='#FFFFFF', ambient=0.2, diffuse=0.8, specular=0.8, specular_power=5, smooth_shading=True)

            # Post
            for x in [-3.05, 3.05]:
                wlh = (0.05, 0.05, 1.55)
                center = np.array([x, 0, wlh[2] * 0.5])
                post_mesh = pyvista.Cube(center, *wlh)
                self.pl.add_mesh(post_mesh, color='#BD7427', ambient=0.2, diffuse=0.8, specular=0.8, specular_power=5, smooth_shading=True)
            
            # Net
            wlh = (6.1, 0.01, 0.79)
            center = np.array([0, 0, (0.76+1.55)/2])
            net_mesh = pyvista.Cube(center, *wlh)
            net_mesh.active_t_coords *= 1000
            tex = pyvista.numpy_to_texture(make_checker_board_texture('#FFFFFF', '#AAAAAA', width=10))
            self.pl.add_mesh(net_mesh, texture=tex, ambient=0.2, diffuse=0.8, opacity=0.1, smooth_shading=True)
            

        # floor
        wlh = (20, 40, 0.05)
        center = np.array([0, 0, -wlh[2] * 0.5])
        floor_mesh = pyvista.Cube(center, *wlh)
        floor_mesh.points[:, 2] -= 0.01
        self.pl.add_mesh(floor_mesh, color='#769771', ambient=0.2, diffuse=0.8, specular=0.8, specular_power=5, smooth_shading=True)

        self.update_smpl_seq(init_args.get('smpl_seq', None), init_args.get('racket_seq', None))
        self.num_actors = init_args.get('num_actors', self.smpl_joints.shape[0])

        colors = get_color_palette(self.num_actors, colormap='autumn' if self.show_skeleton and not self.show_smpl else 'rainbow')
        if self.show_smpl and self.smpl_verts is not None:
            if self.num_actors == 1:
                colors = ['#ffca3a']
            elif self.num_actors == 2:
                colors = ['#000000', '#000000']
            else:
                colors = get_color_palette(self.num_actors, 'rainbow')
            vertices = self.smpl_verts[0, 0].cpu().numpy()
            if init_args.get('debug_root'):
                # Odd actors are final result, even actors are old result
                self.smpl_actors = [SMPLActor(self.pl, vertices, self.smpl_faces, color='#d00000' if i%2==0 else '#ffca3a') 
                    for i in range(self.num_actors)]
            else:
                self.smpl_actors = [SMPLActor(self.pl, vertices, self.smpl_faces, color=colors[a]) for a in range(self.num_actors)]
        if self.show_skeleton:
            if not self.show_smpl:
                colors = get_color_palette(self.num_actors, colormap='autumn')
            else:
                colors = ['yellow'] * self.num_actors
            self.skeleton_actors = [SkeletonActor(self.pl, self.smpl_joint_parents, bone_color=colors[a]) 
                for a in range(self.num_actors)]
        
        if self.show_racket:
            self.racket_actors = [RacketActor(self.pl, init_args.get('sport')) 
                for _ in range(self.num_actors)]
        
    def update_camera(self, interactive):
        pass
        # root_pos = self.smpl_joints[0, self.fr, 0].cpu().numpy()
        # roll = self.pl.camera.roll
        # view_vec = np.asarray(self.pl.camera.position) - np.asarray(self.pl.camera.focal_point) # (5,0,0)
        # new_focal = np.array([root_pos[0], root_pos[1], 0.8])
        # new_pos = new_focal + view_vec
        # self.pl.camera.up = (0, 0, 1)
        # self.pl.camera.focal_point = new_focal.tolist()
        # self.pl.camera.position = new_pos.tolist()
        # self.pl.camera.roll = roll   # don't set roll

    def update_scene(self):
        super().update_scene()

        if self.show_smpl and self.smpl_verts is not None:
            for i, actor in enumerate(self.smpl_actors):
                if self.smpl_joints[i, self.fr].sum() == 0:
                    actor.set_visibility(False)
                else:
                    actor.update_verts(self.smpl_verts[i, self.fr].cpu().numpy())
                    actor.set_visibility(True)
                    actor.set_opacity(0.5)

        if self.show_skeleton:
            for i, actor in enumerate(self.skeleton_actors):
                if self.smpl_joints[i, self.fr].sum() == 0:
                    actor.set_visibility(False)
                else:
                    actor.update_joints(self.smpl_joints[i, self.fr].cpu().numpy())
                    actor.set_visibility(True)
                    actor.set_opacity(1.0)
        
        if self.show_racket:
            for i, actor in enumerate(self.racket_actors):
                if self.smpl_joints[i, self.fr].sum() == 0:
                    actor.set_visibility(False)
                else:
                    actor.update_racket(self.racket_params[i][self.fr])
                    actor.set_visibility(True)

    def setup_key_callback(self):
        super().setup_key_callback()

        def next_data():
            self.update_smpl_seq()

        def reset_camera():
            self.init_camera()

        self.pl.add_key_event('z', next_data)
        self.pl.add_key_event('t', reset_camera)