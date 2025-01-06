import torch
import lie_algebra as Lie


def _cubic_hermite_interp(t0, y0, f0, t1, y1, f1, t):
        h = (t - t0) / (t1 - t0)
        h00 = (1 + 2 * h) * (1 - h) * (1 - h)
        h10 = h * (1 - h) * (1 - h)
        h01 = h * h * (3 - 2 * h)
        h11 = h * h * (h - 1)
        dt = (t1 - t0)
        return h00 * y0 + h10 * dt * f0 + h01 * y1 + h11 * dt * f1


def get_discrete_coeffs(time: torch.Tensor, data: torch.Tensor)->torch.Tensor:
    return data

class DiscreteSpline:
    def __init__(self, times, coeffs, **kwargs):
        super().__init__(**kwargs)
        self._data = coeffs
        self._times = times

    def _interpret_t(self, t):
        assert t >= self._times[0], "t must be greater than or equal to the first time in the spline"
        maxlen = self._data.size(-2) - 1
        index = (t >= self._times).sum() - 1
        index = index.clamp(0, maxlen)  # clamp because t may go outside of [t[0], t[-1]]; this is fine
        # will never access the last element of self._times; this is correct behaviour
        fractional_part = t - self._times[index]
        return fractional_part, index
    
    def evaluate(self, t):
        fractional_part, index = self._interpret_t(t)
        return self._data[...,index,:]
        


def get_cubic_hermite_coeffs(time: torch.Tensor, data: torch.Tensor)->torch.Tensor:
    """ time: torch.Tensor of shape (n,); data: torch.Tensor of shape (...,n, d) """
    """return a, b, c, d of shape (..., n, d) , where a + b * t + c * t^2 + d * t^3 = data (1-d case see CubicHermiteSpline class for more details)"""
    deriv = torch.zeros_like(data)
    deriv[..., 1:, :] = (data[..., 1:, :] - data[..., :-1, :]) / (time[1:] - time[:-1]).unsqueeze(-1) # backward difference

    # hermite cubic spline
    time_tmp = (time[1:] - time[:-1]).unsqueeze(-1)
    a = data[..., :-1, :]
    # b = deriv[..., :-1, :]
    # time_tmp_2 = time_tmp * time_tmp
    # time_tmp_3 = time_tmp * time_tmp_2
    # c = 3 * (data[..., 1:, :] - data[..., :-1, :]) / time_tmp_2 - (2 * deriv[..., :-1, :] + deriv[..., 1:, :]) / time_tmp
    # d = 2 * (data[..., :-1, :] - data[..., 1:, :]) / time_tmp_3 + (deriv[..., :-1, :] + deriv[..., 1:, :]) / time_tmp_2
    b = deriv[..., :-1, :] * time_tmp
    c = 3 * (data[..., 1:, :] - data[..., :-1, :]) - (2 * deriv[..., :-1, :] + deriv[..., 1:, :]) * time_tmp
    d = 2 * (data[..., :-1, :] - data[..., 1:, :]) + (deriv[..., :-1, :] + deriv[..., 1:, :]) * time_tmp
    return a, b, c, d




class CubicHermiteSpline:
    """Calculates the hermite cubic spline approximation to the batch of controls given. Also calculates its derivative.

    Example:
        times = torch.linspace(0, 1, 7)
        # (2, 1) are batch dimensions. 7 is the time dimension (of the same length as t). 3 is the channel dimension.
        X = torch.rand(2, 1, 7, 3)
        coeffs = get_cubic_hermite_coeffs(times, X)
        # ...at this point you can save the coeffs, put them through PyTorch's Datasets and DataLoaders, etc...
        spline = NaturalCubicSpline(times, coeffs)
        t = torch.tensor(0.4)
        # will be a tensor of shape (2, 1, 3), corresponding to batch and channel dimensions
        out = spline.derivative(t)
    """

    def __init__(self, times, coeffs, device = 'cpu', **kwargs):
        """
        Arguments:
            times: As was passed as an argument to natural_cubic_spline_coeffs.
            coeffs: As returned by natural_cubic_spline_coeffs.
        """
        super().__init__(**kwargs)

        a, b, c, d = coeffs

        self._times = times.to(device)
        self._a = a.to(device)
        self._b = b.to(device)
        self._c = c.to(device)
        self._d = d.to(device)
        self._device = device

    def check_t_dim(self, t):
        if t.dim() == 0:
            return True
        elif t.dim() == 1:
            return False
        else:
            raise ValueError("t must be a scalar or a 1D tensor of size 1.")

    def _interpret_t(self, t: torch.Tensor):
        single_t = self.check_t_dim(t)
        if single_t:
            # assert t >= self._times[0], "t must be greater than or equal to the first time in the spline"
            maxlen = self._b.size(-2) - 1
            index = (t > self._times).sum() - 1
            index = index.clamp(0, maxlen)  # clamp because t may go outside of [t[0], t[-1]]; this is fine
            # will never access the last element of self._times; this is correct behaviour
            fractional_part = t - self._times[index]
        else: # batched
            maxlen = self._b.size(-2) - 1
            # t = t.unsqueeze(-1) # (batch_size, 1)
            index = (t.unsqueeze(-1) >= self._times).sum(dim=-1) - 1 # (batch_size,)
            index = index.clamp(0, maxlen)
            fractional_part = t - self._times[index] # (batch_size,)
        return fractional_part, index

    def evaluate(self, t):
        single_t = self.check_t_dim(t)
        if single_t:
            """Evaluates the natural cubic spline interpolation at a point t, which should be a scalar tensor."""
            fractional_part, index = self._interpret_t(t)
            fractional_part = fractional_part / (self._times[index + 1] - self._times[index])
            inner = self._c[..., index, :] + self._d[..., index, :] * fractional_part
            inner = self._b[..., index, :] + inner * fractional_part
            return self._a[..., index, :] + inner * fractional_part
        else:
            fractional_part, index = self._interpret_t(t)
            fractional_part = fractional_part / (self._times[index + 1] - self._times[index])
            fractional_part = fractional_part.unsqueeze(-1)
            if self._a[..., index, :].dim() == 3:
                fractional_part = fractional_part.unsqueeze(0)
            fractional_part = fractional_part.expand(*self._a[..., index, :].shape)
            inner = self._c[..., index, :] + self._d[..., index, :] * fractional_part
            inner = self._b[..., index, :] + inner * fractional_part
            return self._a[..., index, :] + inner * fractional_part

    def derivative(self, t):
        single_t = self.check_t_dim(t)
        if single_t:
            """Evaluates the derivative of the natural cubic spline at a point t, which should be a scalar tensor."""
            fractional_part, index = self._interpret_t(t)
            fractional_part = fractional_part / (self._times[index + 1] - self._times[index])
            inner = 3 * self._d[..., index, :] * fractional_part + self._c[..., index, :] * 2
            return (self._b[..., index, :] + inner * fractional_part) / (self._times[index + 1] - self._times[index])
        else:
            fractional_part, index = self._interpret_t(t)
            tmp = (self._times[index + 1] - self._times[index])
            fractional_part = fractional_part / tmp
            fractional_part = fractional_part.unsqueeze(-1)
            tmp = tmp.unsqueeze(-1)
            if self._a[..., index, :].dim() == 3:
                fractional_part = fractional_part.unsqueeze(0)
                tmp = tmp.unsqueeze(0)
            fractional_part = fractional_part.expand(*self._a[..., index, :].shape)
            inner = 3 * self._d[..., index, :] * fractional_part + self._c[..., index, :] * 2
            return (self._b[..., index, :] + inner * fractional_part) / tmp

class SO3LinearInterpolation:
    def __init__(self, R_time, R_batch, device = 'cpu', **kwargs):
        """
        Arguments:
            times: As was passed as an argument to natural_cubic_spline_coeffs.
            coeffs: As returned by natural_cubic_spline_coeffs.
        """
        super().__init__(**kwargs)

        self._R = R_batch.to(device) # (batch_size, time, 3, 3)
        self._times = R_time.to(device) # (time,)
    
    def set_R(self, R_batch):
        self._R = R_batch # (batch_size, time, 3, 3)

    def check_t_dim(self, t):
        if t.dim() == 0:
            return True
        elif t.dim() == 1:
            return False
        else:
            raise ValueError("t must be a scalar or a 1D tensor of size 1.")

    def _interpret_t(self, t: torch.Tensor):
        single_t = self.check_t_dim(t)
        if single_t:
            # assert t >= self._times[0], "t must be greater than or equal to the first time in the spline"
            maxlen = self._times.shape[0] - 1
            index = (t > self._times).sum() - 1
            index = index.clamp(0, maxlen)  # clamp because t may go outside of [t[0], t[-1]]; this is fine
            # will never access the last element of self._times; this is correct behaviour
            fractional_part = t - self._times[index]
        else: # batched
            raise NotImplementedError("batched evaluation not implemented yet")
        return fractional_part, index
    
    def _SO3_linear_interpolation(self, R0, R1, t0, t1, delta_t):
        """R0, R1: (batch_size, 3, 3)"""
        if abs(delta_t) < 1e-8:
            return R0
        elif abs(delta_t - (t1 - t0)) < 1e-8:
            return R1
        else:
            delta_xi = Lie.SO3log(R1 @ R0.transpose(-1, -2))
            R = R0 @ Lie.SO3exp(delta_xi * (delta_t / (t1 - t0)))
            return R
    
    def evaluate(self, t):
        single_t = self.check_t_dim(t)
        if single_t:
            """Evaluates the natural cubic spline interpolation at a point t, which should be a scalar tensor."""
            fractional_part, index = self._interpret_t(t)
            R0 = self._R[..., index, :, :]
            R1 = self._R[..., index + 1, :, :]
            t0 = self._times[index]
            t1 = self._times[index + 1]
            return self._SO3_linear_interpolation(R0, R1, t0, t1, fractional_part)
            
        else:
            raise NotImplementedError("batched evaluation not implemented yet")


if __name__ == '__main__':
    # print("test cubic hermite spline")
    # time = torch.tensor([0, 2, 4, 6, 8.0,9])
    # data = torch.tensor([[1,2,-2,3.0,5,-3]]).transpose(0,1)
    # # data = data.unsqueeze(0).repeat(2, 1, 1)
    # a, b, c, d = get_cubic_hermite_coeffs(time, data)
    # spline = CubicHermiteSpline(time, (a, b, c, d))
    # t = torch.tensor([0.00])
    # t = torch.tensor(0.0)
    # print("evaluate: ", spline.evaluate(t), "shape", spline.evaluate(t).shape)
    # print("derivative: ", spline.derivative(t), "shape", spline.derivative(t).shape)

    # data_test = torch.rand(5,10,3) # batch_size, time_series, 3
    # time_test = torch.linspace(0, 5, 10)

    # coeff = get_cubic_hermite_coeffs(time_test, data_test)
    # spline = CubicHermiteSpline(time_test, coeff)

    # N = 1000
    # data_plot = torch.empty(5, N, 3)
    # data_derivative_plot = torch.empty(5, N, 3)
    # data_derivatve_difference = torch.empty(5, N, 3)
    # time_plot = torch.linspace(time_test[0]-1, time_test[-1]+1, N)
    # for i, t in enumerate(time_plot):
    #     data_plot[:, i,:] = spline.evaluate(t)
    #     data_derivative_plot[:, i,:] = spline.derivative(t)
    #     # numerical derivative
    #     dt = 1e-6
    #     data_derivatve_difference[:, i,:] = (spline.evaluate(t + dt) - spline.evaluate(t)) / ( dt)

    # from matplotlib import pyplot as plt
    # index_plot = 0 # batch index
    # fig, ax = plt.subplots(3)
    # for i in range(3):
    #     ax[i].plot(time_test.cpu().numpy(), data_test[index_plot,:,i].cpu().numpy(), label="gt")
    #     ax[i].plot(time_plot.cpu().numpy(), data_plot[index_plot,:,i].cpu().numpy(), label="interpolated")
    #     ax[i].legend()
    # fig.suptitle("Cubic Hermite Interpolation")
    # plt.show(block=False)

    # fig, ax = plt.subplots(3)
    # for i in range(3):
    #     ax[i].plot(time_plot.cpu().numpy(), data_derivative_plot[index_plot,:,i].cpu().numpy(), label="derivative")
    #     ax[i].plot(time_plot.cpu().numpy(), data_derivatve_difference[index_plot,:,i].cpu().numpy(), label="numerical derivative")
    #     ax[i].legend()
    # fig.suptitle("Cubic Hermite Derivative")
    # plt.show(block=False)

    # print("test discrete spline")
    # time = torch.tensor([0, 2, 4, 8.0])
    # data = torch.tensor([[1,2,-2,3.0]]).transpose(0,1)
    # data = data.unsqueeze(0).repeat(2, 1, 1)
    # print("data.shape: ", data.shape)
    # spline = DiscreteSpline(time, data)
    # t = torch.tensor([8.0])
    # print("evaluate: ", spline.evaluate(t))
    
    # plt.show()

    import matplotlib.pyplot as plt

    R_batch = Lie.SO3exp(torch.rand(5, 3) * 0.1).unsqueeze(0)
    R_time = torch.linspace(0, 4, 5)
    LinearSO3spline = SO3LinearInterpolation(R_time, R_batch)
    N = 101
    R_interp = torch.zeros(1, N, 3, 3)
    t_plot = torch.linspace(0, 4, N)
    for i, t in enumerate(t_plot):
        R_interp[:, i, :, :] = LinearSO3spline.evaluate(t)
    print("R_batch[-1]", R_batch[0,-1])
    print("R_interp[-1]", R_interp[0,-1])
    plt.figure()
    plt.plot(R_time.cpu().numpy(), R_batch[0,:,0,0].cpu().numpy(), label="gt")
    plt.plot(t_plot.cpu().numpy(), R_interp[0,:,0,0].cpu().numpy(), label="interpolated")
    plt.legend()
    plt.show()
    
        