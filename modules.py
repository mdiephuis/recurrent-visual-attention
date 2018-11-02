import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from torch.autograd import Variable
from helpers.layers import get_encoder, str_to_activ, str_to_activ_module

import numpy as np


class retina(object):
    """
    A retina that extracts a foveated glimpse `phi`
    around location `l` from an image `x`. It encodes
    the region around `l` at a high-resolution but uses
    a progressively lower resolution for pixels further
    from `l`, resulting in a compressed representation
    of the original image `x`.

    Args
    ----
    - x: a 4D Tensor of shape (B, H, W, C). The minibatch
      of images.
    - l: a 2D Tensor of shape (B, 2). Contains normalized
      coordinates in the range [-1, 1].
    - g: size of the first square patch.
    - k: number of patches to extract in the glimpse.
    - s: scaling factor that controls the size of
      successive patches.

    Returns
    -------
    - phi: a 5D tensor of shape (B, k, g, g, C). The
      foveated glimpse of the image.
    """
    def __init__(self, g, k, s):
        self.g = g
        self.k = k
        self.s = s

    def foveate(self, x, l):
        """
        Extract `k` square patches of size `g`, centered
        at location `l`. The initial patch is a square of
        size `g`, and each subsequent patch is a square
        whose side is `s` times the size of the previous
        patch.

        The `k` patches are finally resized to (g, g) and
        concatenated into a tensor of shape (B, k, g, g, C).
        """
        phi = []
        size = self.g

        # extract k patches of increasing size
        for i in range(self.k):
            phi.append(self.extract_patch(x, l, size))
            size = int(self.s * size)

        # resize the patches to squares of size g
        for i in range(1, len(phi)):
            k = phi[i].shape[-1] // self.g
            phi[i] = F.avg_pool2d(phi[i], k)

        # concatenate into a single tensor and flatten
        #print("blaphi=", [p.size() for p in phi])
        phi = torch.cat(phi, 1)
        #phi = phi.view(phi.shape[0], -1)

        return phi

    # iterate over batch
    # This can be paralized, by generating the coordinates and cropping in parallel
    # multi index torch tensor.
    #
    # look at saccade.py CropLambdaPool
    def extract_patch(self, x, l, size):
        """
        Extract a single patch for each image in the
        minibatch `x`.

        Args
        ----
        - x: a 4D Tensor of shape (B, H, W, C). The minibatch
          of images.
        - l: a 2D Tensor of shape (B, 2).
        - size: a scalar defining the size of the extracted patch.

        Returns
        -------
        - patch: a 4D Tensor of shape (B, size, size, C)
        """
        B, C, H, W = x.shape

        # denormalize coords of patch center
        coords = self.denormalize(H, l)

        # compute top left corner of patch
        patch_x = coords[:, 0] - (size // 2)
        patch_y = coords[:, 1] - (size // 2)

        # loop through mini-batch and extract
        patch = []
        for i in range(B):
            im = x[i].unsqueeze(dim=0)
            T = im.shape[-1]

            # compute slice indices
            from_x, to_x = patch_x[i], patch_x[i] + size
            from_y, to_y = patch_y[i], patch_y[i] + size

            # cast to ints
            from_x, to_x = from_x.item(), to_x.item()
            from_y, to_y = from_y.item(), to_y.item()

            # pad tensor in case exceeds
            if self.exceeds(from_x, to_x, from_y, to_y, T):
                pad_dims = (
                    size//2+1, size//2+1,
                    size//2+1, size//2+1,
                    0, 0,
                    0, 0,
                )
                im = F.pad(im, pad_dims, "constant", 0)

                # add correction factor
                from_x += (size//2+1)
                to_x += (size//2+1)
                from_y += (size//2+1)
                to_y += (size//2+1)

            # and finally extract
            patch.append(im[:, :, from_y:to_y, from_x:to_x])

        # concatenate into a single tensor
        patch = torch.cat(patch)

        return patch

    def denormalize(self, T, coords):
        """
        Convert coordinates in the range [-1, 1] to
        coordinates in the range [0, T] where `T` is
        the size of the image.
        """
        return (0.5 * ((coords + 1.0) * T)).long()

    def exceeds(self, from_x, to_x, from_y, to_y, T):
        """
        Check whether the extracted patch will exceed
        the boundaries of the image of size `T`.
        """
        if (
            (from_x < 0) or (from_y < 0) or (to_x > T) or (to_y > T)
        ):
            return True
        return False


class glimpse_network(nn.Module):
    """
    A network that combines the "what" and the "where"
    into a glimpse feature vector `g_t`.

    - "what": glimpse extracted from the retina.
    - "where": location tuple where glimpse was extracted.

    Concretely, feeds the output of the retina `phi` to
    a fc layer and the glimpse location vector `l_t_prev`
    to a fc layer. Finally, these outputs are fed each
    through a fc layer and their sum is rectified.

    In other words:

        `g_t = relu( fc( fc(l) ) + fc( fc(phi) ) )`

    Args
    ----
    - h_g: hidden layer size of the fc layer for `phi`.
    - h_l: hidden layer size of the fc layer for `l`.
    - g: size of the square patches in the glimpses extracted
      by the retina.
    - k: number of patches to extract per glimpse.
    - s: scaling factor that controls the size of successive patches.
    - c: number of channels in each image.
    - x: a 4D Tensor of shape (B, H, W, C). The minibatch
      of images.
    - l_t_prev: a 2D tensor of shape (B, 2). Contains the glimpse
      coordinates [x, y] for the previous timestep `t-1`.

    Returns
    -------
    - g_t: a 2D tensor of shape (B, hidden_size). The glimpse
      representation returned by the glimpse network for the
      current timestep `t`.
    """
    def __init__(self, h_g, h_l, g, k, s, c, config):
        super(glimpse_network, self).__init__()
        self.config = config
        self.retina = retina(g, k, s)
        self.activation_fn = str_to_activ(self.config['activation'])
        activation_module = str_to_activ_module(self.config['activation'])

        # glimpse layer
        self.phi_net = get_encoder(config, name='phi_proj')(
            input_shape=[c, g, g],
            output_size=h_g,
            activation_fn=activation_module
        )
        #D_in = k*g*g*c
        #self.fc1 = nn.Linear(D_in, h_g)

        # location layer
        D_in = 2
        self.loc_net = self._get_dense_net_map(name='loc_proj')(
            D_in, h_l,
            activation_fn=activation_module,
            nlayers=2
        )
        #self.fc2 = nn.Linear(D_in, h_l)

        self.what_net = self._get_dense_net_map(name='what_proj')(
            h_g, h_g + h_l,
            activation_fn=activation_module,
            nlayers=2
        )
        #self.fc3 = nn.Linear(h_g, h_g+h_l)

        self.where_net = self._get_dense_net_map(name='where_proj')(
            h_l, h_g+h_l,
            activation_fn=activation_module,
            nlayers=2
        )
        #self.fc4 = nn.Linear(h_l, h_g+h_l)

    def _get_dense_net_map(self, name):
        '''helper to pull a dense encoder'''
        config = deepcopy(self.config)
        config['encoder_layer_type'] = 'dense'
        return get_encoder(config, name=name)

    def forward(self, x, l_t_prev):
        # phi =  torch.Size([1280, 1600])
        # l_t_prev =  torch.Size([1280, 2])
        # phi_out =  torch.Size([1280, 128])
        # l_out =  torch.Size([1280, 128])
        # what =  torch.Size([1280, 256])
        # where =  torch.Size([1280, 256])
        # g_t =  torch.Size([1280, 256])

        # generate glimpse phi from image x
        phi = self.retina.foveate(x, l_t_prev)
        #print("phi = ", phi.shape)

        # flatten location vector
        l_t_prev = l_t_prev.view(l_t_prev.size(0), -1)

        # feed phi to a conv stack and l to a dense net
        phi_out = self.activation_fn(self.phi_net(phi))
        #print("phi_out = ", phi_out.shape)
        l_out = self.activation_fn(self.loc_net(l_t_prev))
        #print("l_out = ", l_out.shape)

        what = self.what_net(phi_out)
        where = self.where_net(l_out)
        # print("what = ", what.shape)
        # print("where = ", where.shape)

        # feed to fc layer
        g_t = self.activation_fn(what + where)
        # print("g_t = ", g_t.shape)

        # return phi for visualization
        return g_t, phi


class core_network(nn.Module):
    """
    An RNN that maintains an internal state that integrates
    information extracted from the history of past observations.
    It encodes the agent's knowledge of the environment through
    a state vector `h_t` that gets updated at every time step `t`.

    Concretely, it takes the glimpse representation `g_t` as input,
    and combines it with its internal state `h_t_prev` at the previous
    time step, to produce the new internal state `h_t` at the current
    time step.

    In other words:

        `h_t = relu( fc(h_t_prev) + fc(g_t) )`

    Args
    ----
    - input_size: input size of the rnn.
    - hidden_size: hidden size of the rnn.
    - g_t: a 2D tensor of shape (B, hidden_size). The glimpse
      representation returned by the glimpse network for the
      current timestep `t`.
    - h_t_prev: a 2D tensor of shape (B, hidden_size). The
      hidden state vector for the previous timestep `t-1`.

    Returns
    -------
    - h_t: a 2D tensor of shape (B, hidden_size). The hidden
      state vector for the current timestep `t`.
    """
    def __init__(self, input_size, hidden_size):
        super(core_network, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=1)

        # self.i2h = nn.Linear(input_size, hidden_size)
        # self.h2h = nn.Linear(hidden_size, hidden_size)

    def forward(self, g_t, h_t_prev):
        # h1 = self.i2h(g_t)
        # h2 = self.h2h(h_t_prev)
        # h_t = F.relu(h1 + h2)


        # LSTM accepts input, (h_0, c_0)
        # LSTM returns output, (h_n, c_n)
        _, (h_t, c_t) = self.rnn(g_t.unsqueeze(0), h_t_prev)
        return (h_t, c_t)


class action_network(nn.Module):
    """
    Uses the internal state `h_t` of the core network to
    produce the final output classification.

    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a softmax to create a vector of
    output probabilities over the possible classes.

    Hence, the environment action `a_t` is drawn from a
    distribution conditioned on an affine transformation
    of the hidden state vector `h_t`, or in other words,
    the action network is simply a linear softmax classifier.

    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - h_t: the hidden state vector of the core network for
      the current time step `t`.

    Returns
    -------
    - a_t: output probability vector over the classes.
    """
    def __init__(self, input_size, output_size, config):
        super(action_network, self).__init__()
        self.config = config
        self.fc = self._get_dense_net_map('action_proj')(
            input_size, output_size,
            activation_fn=str_to_activ_module(self.config['activation']),
            nlayers=2
        )
        #self.fc = nn.Linear(input_size, output_size)

    def _get_dense_net_map(self, name):
        '''helper to pull a dense encoder'''
        config = deepcopy(self.config)
        config['encoder_layer_type'] = 'dense'
        return get_encoder(config, name=name)

    def forward(self, h_t):
        (h_i, c_i) = h_t
        a_t = F.log_softmax(self.fc(h_i), dim=1)
        return a_t


class location_network(nn.Module):
    """
    Uses the internal state `h_t` of the core network to
    produce the location coordinates `l_t` for the next
    time step.

    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a tanh to clamp the output beween
    [-1, 1]. This produces a 2D vector of means used to
    parametrize a two-component Gaussian with a fixed
    variance from which the location coordinates `l_t`
    for the next time step are sampled.

    Hence, the location `l_t` is chosen stochastically
    from a distribution conditioned on an affine
    transformation of the hidden state vector `h_t`.

    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - std: standard deviation of the normal distribution.
    - h_t: the hidden state vector of the core network for
      the current time step `t`.

    Returns
    -------
    - mu: a 2D vector of shape (B, 2).
    - l_t: a 2D vector of shape (B, 2).
    """
    def __init__(self, input_size, output_size, std, config):
        super(location_network, self).__init__()
        self.std = std
        self.config = config
        self.fc = self._get_dense_net_map('loc_rnn_proj')(
            input_size, output_size,
            activation_fn=str_to_activ_module(self.config['activation']),
            nlayers=2
        )
        #self.fc = nn.Linear(input_size, output_size)

    def _get_dense_net_map(self, name):
        '''helper to pull a dense encoder'''
        config = deepcopy(self.config)
        config['encoder_layer_type'] = 'dense'
        return get_encoder(config, name=name)

    def forward(self, h_t):
        # de-structure the state
        (h_i, c_i) = h_t

        # compute mean
        mu = F.tanh(self.fc(h_i.detach()))

        # reparametrization trick
        noise = torch.zeros_like(mu)
        noise.data.normal_(std=self.std)
        l_t = mu + noise

        # bound between [-1, 1]
        l_t = F.tanh(l_t)

        return mu, l_t


class baseline_network(nn.Module):
    """
    Regresses the baseline in the reward function
    to reduce the variance of the gradient update.

    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - h_t: the hidden state vector of the core network
      for the current time step `t`.

    Returns
    -------
    - b_t: a 2D vector of shape (B, 1). The baseline
      for the current time step `t`.
    """
    def __init__(self, input_size, output_size, config):
        super(baseline_network, self).__init__()
        self.config = config
        self.fc = self._get_dense_net_map('baseline_proj')(
            input_size, output_size,
            activation_fn=str_to_activ_module(self.config['activation']),
            nlayers=2
        )
        #self.fc = nn.Linear(input_size, output_size)

    def _get_dense_net_map(self, name):
        '''helper to pull a dense encoder'''
        config = deepcopy(self.config)
        config['encoder_layer_type'] = 'dense'
        return get_encoder(config, name=name)

    def forward(self, h_t):
        (h_i, c_i) = h_t
        #b_t = str_to_activ(self.config['activation'])(self.fc(h_i.detach()))
        b_t = str_to_activ(self.config['activation'])(self.fc(h_i))
        return b_t
