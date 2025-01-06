from SO3diffeq import odeint_SO3
from Rigid_body import *
import matplotlib.pyplot as plt
import BiasDy.lie_algebra as Lie

import time

if __name__ == "__main__":   
    print("------------------------------------")
    print("Test IMU integration")
    print("------------------------------------")

    ## debug setting
    plotflag = True
    torch.autograd.set_detect_anomaly(False)

    torch.set_printoptions(precision=5, sci_mode=False)
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    ## generate true data
    """angular velocity and SO3"""
    torch.manual_seed(0)
    tw1 = screw_axis_from_joint(torch.tensor([1., 0, 0]), torch.rand(3), 'revolute')
    tw2 = screw_axis_from_joint(torch.tensor([0., 1, 0]), torch.rand(3), 'revolute')
    tw3 = screw_axis_from_joint(torch.tensor([0., 0, 1]), torch.rand(3), 'revolute')
    tw_all = torch.cat((tw1.unsqueeze(1), tw2.unsqueeze(1), tw3.unsqueeze(1)), dim=1).to(device)
    theta_all = lambda t: 3 * torch.tensor([torch.sin(t), torch.sin(t), torch.sin(t)]).to(device)
    theta_all_dot = lambda t: 3 * torch.tensor([torch.cos(t), torch.cos(t), torch.cos(t)]).to(device)

    gst0 = torch.eye(4).to(device)
    gst0[:3, :3] = SO3exp_from_unit_vec(torch.tensor([0., 1.0, 0]), torch.tensor(0.5))
    gst0[:3, :3] = Lie.SO3exp(torch.rand(3))
    gst0[:3, 3] = torch.tensor([1., 2, 3])

    wb_true = lambda t: (Body_Jacobian(tw_all, theta_all(t), gst0) @ theta_all_dot(t))[:3]
    R_true = lambda t: forward_kinematics(tw_all, theta_all(t), gst0)[:3, :3]

    wb_true_batch = lambda t: torch.stack([wb_true(t) for t in t])
    R_true_batch = lambda t: torch.stack([R_true(t) for t in t])

    """v, p and a"""
    p_true = lambda t: torch.tensor([torch.sin(t), torch.cos(t), torch.sin(t)]).to(device)
    v_true = lambda t: torch.tensor([torch.cos(t), -torch.sin(t), torch.cos(t)]).to(device)
    dot_v_true = lambda t: torch.tensor([-torch.sin(t), -torch.cos(t), -torch.sin(t)]).to(device)
    # \dot v = R a + g -> a = R^T (\dot v - g)
    g_const = torch.tensor([0, 0, -9.81]).to(device)
    a_true = lambda t: R_true(t).T @ (dot_v_true(t) - g_const)
    a_true_batch = lambda t: torch.stack([a_true(t) for t in t])
    ## numerical check
    # t = torch.linspace(0, 20, 1000).to(device)
    # dt = t[1] - t[0]
    # wb_true_all = torch.stack([wb_true(t) for t in t])
    # R_true_all = torch.stack([R_true(t) for t in t])
    # wb_n = Lie.so3vee(R_true_all[1:].transpose(-1,-2) @ (R_true_all[1:] - R_true_all[:-1]) / dt )
    # wb_n = Lie.SO3log(R_true_all[:-1].transpose(-1,-2) @ R_true_all[1:]) / dt
    # plt.plot(t[1:], wb_true_all[1:], label="true")
    # plt.plot(t[1:], wb_n, label="numerical")
    # plt.legend()
    # plt.show()
    # delta_xi = Lie.SO3log(R_true_all[0].transpose(-1,-2) @ R_true_all)
    # plt.plot(t, delta_xi.norm(dim=-1))
    # plt.show()

    ## Euler integration
    t_int = torch.linspace(0, 20, 256+1).to(device)
    dt = t_int[1] - t_int[0]
    R_true_all = torch.stack([R_true(t) for t in t_int])
    R_Euler = torch.zeros_like(R_true_all)
    v_Euler = torch.zeros((len(t_int), 3)).to(device)
    p_Euler = torch.zeros((len(t_int), 3)).to(device)
    R_Euler[0] = R_true_all[0]
    v_Euler[0] = v_true(t_int[0])
    p_Euler[0] = p_true(t_int[0])
    for i in range(1, len(t_int)):
        R_Euler[i] = R_Euler[i-1] @ Lie.SO3exp(wb_true(t_int[i-1]) * dt)
        v_Euler[i] = v_Euler[i-1] + (R_Euler[i-1] @ a_true(t_int[i-1]).unsqueeze(-1)).squeeze(-1) * dt + g_const * dt
        p_Euler[i] = p_Euler[i-1] + v_Euler[i-1] * dt 
    error_Euler = (R_Euler - R_true_all).norm(dim=(-2,-1)) # shape (1001,)
    
    
    ####### ONLY SO3 integration #######
    print("------------------------------------")
    print("Test SO3 integration")
    print("------------------------------------")

    print("Conventional_Euler error: ", (R_Euler - R_true_all).norm().item())

    class ODE_SO3(torch.nn.Module):
        def forward(self, t, y):
            # dy = (Lie.SO3leftJacoInv(y[...,:3]) @ ws_true(t).unsqueeze(1)).squeeze(1)
            dy = (Lie.SO3rightJacoInv(y[...,:3]) @ wb_true(t).unsqueeze(1)).squeeze(1)
            return dy
    func = ODE_SO3().to(device)
    y0 = torch.zeros(3).to(device)

    # for method in ['dopri5', 'euler', 'rk4', 'dopri8', 'bosh3', 'fehlberg2', 'adaptive_heun', 'midpoint', 'heun3']:
    for method in ['dopri5', 'euler', 'rk4']:
        solution, R_sol = odeint_SO3(func, y0, R_true_all[0], t_int, options=None, method=method) # dopri5
        print("SO3odeint error with method %s: " % method, (R_sol - R_true_all).norm().item())
        error_dopri5 = (R_sol - R_true_all).norm(dim=(-2,-1)) # shape (1001,)
        if plotflag:
            plt.figure()
            plt.plot(t_int, error_Euler, label="Conventional_Euler")
            plt.plot(t_int, error_dopri5.detach().numpy(), label="SO3odeint")
            plt.legend()
            plt.title("Error of Conventional Euler and SO3odeint with method: %s" % method)
            plt.show(block=False)
        

        ## test backward
        class ODE_SO3_network(torch.nn.Module):
            def __init__(self):
                super(ODE_SO3_network, self).__init__()
                self.fc1 = torch.nn.Linear(3, 3)
            def forward(self, t, y):
                # dy = (Lie.SO3leftJacoInv(y[...,:3]) @ ws_true(t).unsqueeze(1)).squeeze(1)
                dy = (Lie.SO3rightJacoInv(y[...,:3]) @ self.fc1(wb_true(t)).unsqueeze(1)).squeeze(1)
                return dy
        func_grad = ODE_SO3_network().to(device)
        optimizer = torch.optim.Adam(func_grad.parameters(), lr=1e-2)
        optimizer.zero_grad()
        y0 = torch.zeros(3).to(device)
        solution, R_sol = odeint_SO3(func_grad, y0, R_true_all[0], t_int, method=method) # dopri5
        loss = (R_sol - R_true_all).norm()
        loss.backward()
        optimizer.step()
        print("backward check for method %s passed!" % (method))
    ##########################################

    ## test SE23 integration
    print("------------------------------------")
    print("Test SE23 integration")
    print("------------------------------------")
    class ODE_SE23(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.R0 = torch.eye(3).to(device)
        def forward(self, t, y):
            # dy = (Lie.SO3leftJacoInv(y[...,:3]) @ ws_true(t).unsqueeze(1)).squeeze(1)
            dxi = (Lie.SO3rightJacoInv(y[...,:3]) @ wb_true(t).unsqueeze(1)).squeeze(1)
            dv = (self.R0 @ Lie.SO3exp(y[...,:3]) @ a_true(t).unsqueeze(-1)).squeeze(-1) + g_const.expand_as(a_true(t)).to(device)
            dp = y[...,3:6]
            return torch.cat((dxi, dv, dp), dim=-1)
        def set_R0(self, R: torch.Tensor):
            self.R0 = R
        def callback_change_chart(self, R: torch.Tensor):
            self.set_R0(R)
    func_SE23 = ODE_SE23().to(device)
    func_SE23.set_R0(R_true_all[0])
    t0 = torch.tensor(0.0).to(device)
    y0 = torch.cat([torch.zeros(3).to(device),  v_true(t0), p_true(t0)], dim=-1)
    v_gt = torch.stack([v_true(t) for t in t_int])
    error_v_Euler = (v_Euler - v_gt).norm()
    print("error of v Convention_Euler: ", error_v_Euler.mean().item())
    for method in ['dopri5', 'euler', 'rk4']: #['dopri5', 'euler', 'rk4', 'dopri8', 'bosh3', 'fehlberg2', 'adaptive_heun', 'midpoint', 'heun3']
        solution, R_sol = odeint_SO3(func_SE23, y0, R_true_all[0], t_int, rtol=1e-7, atol=1e-9,method=method) # dopri5
        
        v_pred = solution[..., 3:6]
        p_pred = solution[..., 6:9]
        
        # error_R_Euler = (R_Euler - R_true_all).norm(dim=(-2,-1))
        error_R = (R_sol - R_true_all).norm(dim=(-2,-1))
        error_v = (v_pred - v_gt).norm()
        # print("error of R Euler: ", error_R_Euler.mean().item())
        print("error of R with method %s: " % method, error_R.mean().item())
        print("error of v with method %s: " % method, error_v.mean().item())
        
        if plotflag:
            fig, ax = plt.subplots(3, 1)
            for i in range(3):
                ax[i].plot(t_int, v_gt[...,i], label="v_true")
                ax[i].plot(t_int, v_pred[...,i], label="v_pred")
                ax[i].plot(t_int, v_Euler[...,i], label="v_Euler")
                ax[i].legend()
            fig.suptitle('velocity with method %s' % method)
            plt.show(block=False)

    
        # test the backward
        class ODE_SE23_Grad(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.R0 = torch.eye(3).to(device)
                self.fc1 = torch.nn.Linear(3, 3)
                self.fc2 = torch.nn.Linear(3, 3)

            def forward(self, t, y):
                # dy = (Lie.SO3leftJacoInv(y[...,:3]) @ ws_true(t).unsqueeze(1)).squeeze(1)
                dxi = (Lie.SO3rightJacoInv(y[...,:3]) @ self.fc1(wb_true(t)).unsqueeze(1)).squeeze(1)
                dv = (self.R0 @ Lie.SO3exp(y[...,:3]) @ self.fc2(a_true(t)).unsqueeze(-1)).squeeze(-1) + g_const.expand_as(a_true(t)).to(device)
                dp = y[...,3:6]
                return torch.cat((dxi, dv, dp), dim=-1)
            
            def set_R0(self, R: torch.Tensor):
                self.R0 = R
            def callback_change_chart(self, R: torch.Tensor):
                self.set_R0(R)
        
        func_SE23_grad = ODE_SE23_Grad().to(device)
        func_SE23_grad.set_R0(R_true_all[0])
        optimizer = torch.optim.Adam(func_SE23_grad.parameters(), lr=1e-2)
        optimizer.zero_grad()
        y0 = torch.cat([torch.zeros(3).to(device),  v_true(t0), p_true(t0)], dim=-1)
        solution, R_sol = odeint_SO3(func_SE23_grad, y0, R_true_all[0], t_int, method=method)
        loss = (R_sol - R_true_all).norm() + (solution - torch.rand_like(solution)).norm()
        loss.backward()
        optimizer.step()
        print("backward check for method %s passed!" % (method))

    if plotflag:
        plt.show()

    print("All tests passed!")