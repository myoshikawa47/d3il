#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
# Released under the MIT License.
#

import torch
import torch.nn as nn
from eipl.layer import SpatialSoftmax, InverseSpatialSoftmax


class SARNN(nn.Module):
    #:: SARNN
    """SARNN: Spatial Attention with Recurrent Neural Network.
    This model "explicitly" extracts positions from the image that are important to the task, such as the work object or arm position,
    and learns the time-series relationship between these positions and the robot's joint angles.
    The robot is able to generate robust motions in response to changes in object position and lighting.

    Arguments:
        rec_dim (int): The dimension of the recurrent state in the LSTM cell.
        k_dim (int, optional): The dimension of the attention points.
        action_dim (int, optional): The dimension of the joint angles.
        temperature (float, optional): The temperature parameter for the softmax function.
        heatmap_size (float, optional): The size of the heatmap in the InverseSpatialSoftmax layer.
        kernel_size (int, optional): The size of the convolutional kernel.
        activation (str, optional): The name of activation function.
        im_size (list, optional): The size of the input image [height, width].
    """

    def __init__(
        self,
        rec_dim,
        k_dim=5,
        obs_dim=20,
        action_dim=3,
        temperature=1e-4,
        heatmap_size=0.1,
        kernel_size=3,
        visual_input=False,
        im_size=[64, 64],
    ):
        super(SARNN, self).__init__()

        self.k_dim = k_dim
        activation = nn.LeakyReLU(negative_slope=0.3)
        self.visual_input = visual_input
        
        if self.visual_input:
            sub_im_size = [
                im_size[0] - 3 * (kernel_size - 1),
                im_size[1] - 3 * (kernel_size - 1),
            ]
            self.temperature = temperature
            self.heatmap_size = heatmap_size

            # Positional Encoder
            self.pos_encoder = nn.Sequential(
                nn.Conv2d(3, 16, 3, 1, 0),  # Convolutional layer 1
                activation,
                nn.Conv2d(16, 32, 3, 1, 0),  # Convolutional layer 2
                activation,
                nn.Conv2d(32, self.k_dim, 3, 1, 0),  # Convolutional layer 3
                activation,
                SpatialSoftmax(
                    width=sub_im_size[0],
                    height=sub_im_size[1],
                    temperature=self.temperature,
                    normalized=True,
                ),  # Spatial Softmax layer
            )

            # Image Encoder
            self.im_encoder = nn.Sequential(
                nn.Conv2d(3, 16, 3, 1, 0),  # Convolutional layer 1
                activation,
                nn.Conv2d(16, 32, 3, 1, 0),  # Convolutional layer 2
                activation,
                nn.Conv2d(32, self.k_dim, 3, 1, 0),  # Convolutional layer 3
                activation,
            )
            rec_in = obs_dim + self.k_dim * 2 * 2
        
        else:
            rec_in = obs_dim
        
        self.rec = nn.LSTMCell(rec_in, rec_dim)  # LSTM cell

        # Joint Decoder
        self.decoder_action = nn.Sequential(
            nn.Linear(rec_dim, action_dim), activation
        )  # Linear layer and activation

        if self.visual_input:
            # Point Decoder
            self.decoder_point_agent = nn.Sequential(
                nn.Linear(rec_dim, self.k_dim * 2), activation
            )  # Linear layer and activation
           
            self.decoder_point_hand = nn.Sequential(
                nn.Linear(rec_dim, self.k_dim * 2), activation
            )  # Linear layer and activation

            # Inverse Spatial Softmax
            self.issm = InverseSpatialSoftmax(
                width=sub_im_size[0],
                height=sub_im_size[1],
                heatmap_size=self.heatmap_size,
                normalized=True,
            )

            # Image Decoder
            self.decoder_image = nn.Sequential(
                nn.ConvTranspose2d(
                    self.k_dim, 32, 3, 1, 0
                ),  # Transposed Convolutional layer 1
                activation,
                nn.ConvTranspose2d(32, 16, 3, 1, 0),  # Transposed Convolutional layer 2
                activation,
                nn.ConvTranspose2d(16, 3, 3, 1, 0),  # Transposed Convolutional layer 3
                activation,
            )

        self.apply(self._weights_init)

    def _weights_init(self, m):
        """
        Tensorflow/Keras-like initialization
        """
        if isinstance(m, nn.LSTMCell):
            nn.init.xavier_uniform_(m.weight_ih)
            nn.init.orthogonal_(m.weight_hh)
            nn.init.zeros_(m.bias_ih)
            nn.init.zeros_(m.bias_hh)

        if (
            isinstance(m, nn.Conv2d)
            or isinstance(m, nn.ConvTranspose2d)
            or isinstance(m, nn.Linear)
        ):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, xi, xv, rnn_state=None):
        """
        Forward pass of the SARNN module.
        Predicts the image, joint angle, and attention at the next time based on the image and joint angle at time t.
        Predict the image, joint angles, and attention points for the next state (t+1) based on
        the image and joint angles of the current state (t).
        By inputting the predicted joint angles as control commands for the robot,
        it is possible to generate sequential motion based on sensor information.

        Arguments:
            xi (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width).
            xv (torch.Tensor): Input vector tensor of shape (batch_size, input_dim).
            state (tuple, optional): Initial hidden state and cell state of the LSTM cell.

        Returns:
            y_image (torch.Tensor): Decoded image tensor of shape (batch_size, channels, height, width).
            y_joint (torch.Tensor): Decoded joint prediction tensor of shape (batch_size, action_dim).
            enc_pts (torch.Tensor): Encoded points tensor of shape (batch_size, k_dim * 2).
            dec_pts (torch.Tensor): Decoded points tensor of shape (batch_size, k_dim * 2).
            rnn_hid (tuple): Tuple containing the hidden state and cell state of the LSTM cell.
        """
        if self.visual_input:

            agentview_image, in_hand_image, state = state

            B, C, H, W = agentview_image.size()

            # Encode input image
            im_hid_agent = self.im_encoder(agentview_image)
            im_hid_hand = self.im_encoder(in_hand_image)
            enc_pts_agent, _ = self.pos_encoder(agentview_image)
            enc_pts_hand, _ = self.pos_encoder(in_hand_image)
            
            # Reshape encoded points and concatenate with input vector
            enc_pts_agent = enc_pts_agent.reshape(-1, self.k_dim * 2)
            enc_pts_hand = enc_pts_hand.reshape(-1, self.k_dim * 2)
            hid = torch.cat([enc_pts_agent, enc_pts_hand, xv], -1)
            
        else:
            hid = state

        rnn_hid = self.rec(hid, rnn_state)  # LSTM forward pass
        y_action = self.decoder_action(rnn_hid[0])  # Decode joint prediction
        
        if self.vision_input:
            dec_pts_agent = self.decoder_point_agent(rnn_hid[0])  # Decode points
            dec_pts_hand = self.decoder_point_hand(rnn_hid[0])  # Decode points

            # agent image
            dec_pts_in = dec_pts_agent.reshape(-1, self.k_dim, 2)
            heatmap = self.issm(dec_pts_in)  # Inverse Spatial Softmax
            hid = torch.mul(heatmap, im_hid_agent)  # Multiply heatmap with image feature `im_hid`

            y_image_agent = self.decoder_image(hid)  # Decode image
            
            # in-hand image
            dec_pts_in = dec_pts_hand.reshape(-1, self.k_dim, 2)
            heatmap = self.issm(dec_pts_in)  # Inverse Spatial Softmax
            hid = torch.mul(heatmap, im_hid_hand)  # Multiply heatmap with image feature `im_hid`

            y_image_hand = self.decoder_image(hid)  # Decode image
            
            return y_image_agent, y_image_hand, \
                    y_action, \
                    enc_pts_agent, enc_pts_hand, \
                    dec_pts_agent, dec_pts_hand, \
                    rnn_hid

        else:
            return y_action, rnn_hid