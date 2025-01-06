import torch


def screw_axis_from_joint(direction_axis : torch.Tensor, p_on_axis : None, joint_type = 'revolute' or 'prismatic') -> torch.Tensor:
    """
    Compute the screw axis from the joint
    :param direction_axis: the joint direction should be a unit vector, shape (3,)
           p_on_axis: a point on the axis   shape (3,)
           joint_type: the type of the joint, 'revolute' or 'prismatic'
    :return: the screw axis of the joint, shape (6,)
    """

    if joint_type == 'revolute':
        if p_on_axis is None:
            raise ValueError('A point on the axis is needed for revolute joint')
        return torch.cat((direction_axis, torch.linalg.cross(p_on_axis, direction_axis)))
    elif joint_type == 'prismatic':
        return torch.cat((direction_axis, torch.tensor([0, 0, 0])))
    else:
        raise ValueError('Invalid joint type')
    

def SO3exp_from_unit_vec(vec,theta)->torch.Tensor:
    """
    Compute the exponential of a unit vector in SO(3)
    :param vec: the unit vector, shape (3,)
           theta: the rotation angle
    :return: the exponential of the unit vector in SO(3), shape (3,3)
    """
    skew_symmetric = torch.tensor([[0, -vec[2], vec[1]],
                                   [vec[2], 0, -vec[0]],
                                   [-vec[1], vec[0], 0]]).to(vec.device)
    return torch.eye(3).to(vec.device) + torch.sin(theta) * skew_symmetric + (1 - torch.cos(theta)) * skew_symmetric @ skew_symmetric

def SE3exp_from_unit_twist(twist, theta)->torch.Tensor:
    """
    Compute the exponential of a unit twist in SE(3)
    :param twist: the unit twist, shape (6,), first 3 elements are the rotation axis, last 3 elements are the translation vector
           theta: the rotation angle
    :return: the exponential of the unit twist in SE(3), shape (4,4)
    """
    omega = twist[:3]
    v = twist[3:]
    R = SO3exp_from_unit_vec(omega, theta)
    output = torch.eye(4).to(twist.device)
    if torch.norm(omega) == 0:
        output[:3, 3] = v
        return output
    else:
        output[:3, :3] = R
        output[:3, 3] = (torch.eye(3).to(twist.device) - R) @ torch.linalg.cross(omega, v) + omega * omega @ v * theta
        return output
        

def Adjoint_from_SE3(g)->torch.Tensor:
    """
    Compute the adjoint matrix from a SE(3) matrix
    :param g: the SE(3) matrix, shape (4,4)
    :return: the adjoint matrix, shape (6,6)
    """
    R = g[:3, :3]
    p = g[:3, 3]
    p_hat = torch.tensor([[0, -p[2], p[1]],
                          [p[2], 0, -p[0]],
                          [-p[1], p[0], 0]])
    return torch.cat((torch.cat((R, torch.zeros((3, 3))), dim=1), torch.cat((p_hat @ R, R), dim=1)) , dim=0)

def Adjoint_from_SE3_inv(g)->torch.Tensor:
    """
    Compute the inverse of the adjoint matrix from a SE(3) matrix
    :param g: the SE(3) matrix, shape (4,4)
    :return: the inverse of the adjoint matrix, shape (6,6)
    """
    R = g[:3, :3].transpose(0, 1)
    p = g[:3, 3]
    p_hat = torch.tensor([[0, -p[2], p[1]],
                          [p[2], 0, -p[0]],
                          [-p[1], p[0], 0]]).to(g.device)
    return torch.cat((torch.cat((R, torch.zeros((3, 3)).to(g.device)), dim=1), torch.cat((-R @ p_hat, R), dim=1)) , dim=0)

def Body_Jacobian(twist_all, theta_all,gst0)->torch.Tensor:
    """
    Compute the body Jacobian
    :param twist_all: the screw axes of the joints, shape (6, n)
           theta_all: the joint angles, shape (n,)
           gst0: the initial configuration, shape (4,4)
    :return: the body Jacobian, shape (6, n)
    """
    n = twist_all.shape[1]
    Jb = torch.zeros((6, n)).to(twist_all.device)
    for i in range(n):
        pro_exp = torch.eye(4).to(twist_all.device)
        for j in range(i, n):
            pro_exp = pro_exp @ SE3exp_from_unit_twist(twist_all[:, j], theta_all[j])
        pro_exp = pro_exp @ gst0
        Jb[:, i] = Adjoint_from_SE3_inv(pro_exp) @ twist_all[:, i]
    return Jb

def forward_kinematics(twist_all, theta_all, gst0)->torch.Tensor:
    """
    Compute the forward kinematics
    :param twist_all: the screw axes of the joints, shape (6, n)
           theta_all: the joint angles, shape (n,)
           gst0: the initial configuration, shape (4,4)
    :return: the forward kinematics, shape (4,4)
    """
    n = twist_all.shape[1]
    pro_exp = torch.eye(4).to(twist_all.device)
    for i in range(n):
        pro_exp = pro_exp @ SE3exp_from_unit_twist(twist_all[:, i], theta_all[i])
    return pro_exp @ gst0

if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    torch.set_printoptions(precision=3, sci_mode=False)

    tw1 = screw_axis_from_joint(torch.tensor([1., 0, 0]), torch.tensor([1., 2, 3]), 'revolute')
    tw2 = screw_axis_from_joint(torch.tensor([0., 1, 0]), torch.tensor([4., 5, 6]), 'revolute')
    tw3 = screw_axis_from_joint(torch.tensor([0., 0, 1]), torch.tensor([7., 8, 9]), 'revolute')
    tw4 = screw_axis_from_joint(torch.tensor([1., 0, 0]), torch.tensor([10., 11, 12]), 'revolute')
    tw5 = screw_axis_from_joint(torch.tensor([0., 1, 0]), torch.tensor([13., 14, 15]), 'revolute')
    tw6 = screw_axis_from_joint(torch.tensor([0., 0, 1]), torch.tensor([16., 17, 18]), 'revolute')
    tw_all = torch.cat((tw1.unsqueeze(1), tw2.unsqueeze(1), tw3.unsqueeze(1), tw4.unsqueeze(1), tw5.unsqueeze(1), tw6.unsqueeze(1)), dim=1)
    print("tw_all: \n", tw_all)
    theta_all = torch.tensor([1, 2, 3, 4, 5, 6.0])

    gst0 = torch.eye(4)
    gst0[:3, 3] = torch.tensor([1., 2, 3])
    gst = forward_kinematics(tw_all, theta_all, gst0)
    print("forward kinematics: \n", gst)

    Jb = Body_Jacobian(tw_all, theta_all, gst0)
    print(Jb)

    Js = Adjoint_from_SE3(gst) @ Jb
    print("Js: \n", Js)

    # SE3_test = SE3exp_from_unit_twist(tw3,torch.tensor(0.5))
    # print(SE3_test)

    # Adjoint_test = Adjoint_from_SE3_inv(SE3_test)
    # print(Adjoint_test)