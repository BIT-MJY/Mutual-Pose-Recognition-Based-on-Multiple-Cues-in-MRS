# Developed by Kai
# Reference: 


import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys
from enum import Enum

'''
    Need to import private functions from tensorflow.python.keras instead of tf.keras.backend 
    to get symbolic_learning_phase() to use with K.function an eager execution enabled. 
    Otherwise we will run into an error
    see my entry on https://github.com/tensorflow/tensorflow/issues/34201
'''
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils.conv_utils import convert_data_format

''' Some variables '''
tfk = tf.keras

class EP:

    def __init__(self, model):
        self._use_Gaussian_ReLU = False
        self._model = model
        self._ep_model = None
        self.DEBUG = Enum('DEBUG','model layer plot tensor')
        self._debug = dict({self.DEBUG.model: False,
                            self.DEBUG.layer: False,
                            self.DEBUG.plot: False,
                            self.DEBUG.tensor: None
                           })

        self.init_logger()


    def init_logger(self, log_level=logging.INFO):
        """ Logger
        :param log_level:
        """
        log_format = (
            # '%(asctime)s - '
            # '%(name)s - '
                self.__class__.__name__ +
                '(%(funcName)s) - '
                '%(levelname)s - '
                '%(message)s'
        )

        bold_seq = '\033[1m'
        ep_format = (
            # f'{bold_seq} '
            # '%(log_color)s '
            f'{log_format}'
        )

        ''' If one of the debugs is enabled reset the loglevel'''
        if not all((not value or value is None) for value in self._debug.values()):
            ll = logging.DEBUG
        else:
            ll = log_level

        # Reset the logging state
        logging.getLogger().setLevel(ll)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(stream=sys.stdout, format=ep_format)

    # Own analytical activation function. Input parameters needs to be (input, error)
    @staticmethod
    def gaussian_relu(layer_input, error):
        """ See formula Gaussian ReLU (Equation 44).
        The implementation below is based on the variable names in the paper.

        :param layer_input:
        :param error:
        :return:
        """

        std = tf.sqrt(error)
        mu = layer_input

        # E(Y^2): Term1
        term1 = ((mu ** 2.0 + std ** 2.0) / 2.0) * (tf.math.erf(mu / (std * tf.sqrt(2.0))) + 1.0)

        # E(Y^2): Term2
        term2 = (mu * std * tf.math.exp(-0.5 * (mu / std) ** 2.0)) / tf.sqrt(2.0 * np.pi)

        # E(Y)^2: Term3
        term3 = (std * tf.math.exp(-0.5 * (mu / std) ** 2.0)) / tf.sqrt(2.0 * np.pi)

        # E(Y)^2: Term4
        term4 = (mu / 2.0) * (1.0 - tf.math.erf(-mu / (std * tf.sqrt(2.0))))

        # Expectation
        EY = term3 + term4

        # Squared (outer) Expectation
        EY_2 = EY ** 2

        # Squared (inner) Expectation
        EY2 = term1 + term2

        # Law of Total Variance = Var(Y) = E(Y^2)-E(Y)^2
        VAR = EY2 - EY_2

        # Return the mean and the variance
        return VAR

    @staticmethod
    def softmax_jacobian(layer_input, error):

        """ Because we do not want to change the behavior w.r.t. the other activation derivatives,
        we need to calculate the softmax first, to estimate the derivative.
        We could also take the output of the last affine layer (if activation softmax is used), but then we have
        to change the behavior of the layer propagation in between.
        Long story short, we want to have the same behaviour as the others, thus we calculate the softmax here

        :param layer_input:
        :param error:
        :return:
        """

        # m, n = layer_input.shape
        sm = tf.nn.softmax(layer_input)

        ''' First we create for each example feature vector, it's outer product with itself
            ( p1^2  p1*p2  p1*p3 .... )
            ( p2*p1 p2^2   p2*p3 .... )
            ( ...                     )
        '''
        tensor1 = tf.einsum('ij,ik->ijk', sm, sm)  # (m, n, n)
        ''' Second we need to create an (n,n) identity of the feature vector
            ( p1  0  0  ...  )
            ( 0   p2 0  ...  )
            ( ...            )
        '''
        tensor2 = tf.einsum('ij,jk->ijk', sm, tf.eye(layer_input.shape[-1]))  # (m, n, n)
        ''' Then we need to subtract the first tensor from the second
            ( p1 - p1^2   -p1*p2   -p1*p3  ... )
            ( -p1*p2     p2 - p2^2   -p2*p3 ...)
            ( ...                              )
        '''
        df_dx = tensor2 - tensor1
        df_dx = tf.dtypes.cast(df_dx, tf.float32)

        ''' Finally, we multiply the df_dx with the error (variance) to get the gradient w.r.t. x
            In our case we are interested in calculating the variance. 
        '''
        error = tf.einsum('ijk,ik->ij', df_dx ** 2, error)  # (m, n)

        return error

    def create_EP_Model(self, use_gaussian_relu=False, verbose=True):
        error = None
        layers = None
        self._use_Gaussian_ReLU = use_gaussian_relu

        # check if one of the debug options is enabled
        if all((not value or value is None) for value in self._debug.values()):
            if not verbose:
                self.init_logger(logging.WARNING)
            else:
                self.init_logger(logging.INFO)

        logging.info(f'--------- Start Error Propagation for {len(self._model.layers) - 1} Layers ---------\n')

        for i, l in enumerate(self._model.layers):

            # # Check if only a specific layer shall be debugged
            # if self._debug_layer is not None:
            #     if l.__class__.__name__ == self._debug_layer:
            #         # Call the EP function related to the layer name with the specified tensor
            #         error = getattr(self, str(l.__class__.__name__))(error, l)
            #     else:
            #         continue
            # else:
            # Call the EP function related to the layer name
            error = getattr(self, str(l.__class__.__name__))(error, l)


        ''' Check if error is still None'''
        if error is None:
            # raise RuntimeError('Mhhh something went wrong... Check if there is a dropout layer in model!')
            logging.error('Mhhh something went wrong... Check if there is a dropout layer in model!')
            # set error to zero in this case
            error = tf.zeros_like(self._model.outputs)[0, 0:]

        # if isinstance(error, (np.ndarray)):
        #     error = tf.convert_to_tensor(error, dtype=tf.float32)

        ''' Concatenate Prior model output and error
            Either uses tfk.layers.concatenate() to have one output (e.g. None,2) or add the error
            as a list item [error] and you will get two separate outputs [e.g. (None,1),(None,1)].
            I used the second one because it is more intuitive
        '''
        #ep_model_output = tfk.layers.concatenate([self._model.output, error])
        ep_model_output = self._model.outputs + [error]

        if not self._debug[self.DEBUG.model]:
            ''' Create new deterministic Model with error (variance) output'''
            self._ep_model = tfk.Model(self._model.inputs, ep_model_output)
            return self._ep_model
        else:
            logging.warning('Do not create a model in Debug-Mode')
            self._ep_model = None

    def debug_layer(self, layer_id=None, debug_input_tensor=None, batch_size=None):
        """
        Function for debugging a layer using either an own tensor or it is created a random tensor automatically

        :param layer_id: Id of the model layer which shall be called by the function
        :param debug_input_tensor: Own Tensor for debugging
        :param batch_size: Batch_Size for auto generated tensor. If not specified the batch_size is 1 for higher
        dimensional layer inputs and 10 for one-dimensional inputs
        :return: No return
        """
        assert layer_id is not None, "Please specify a valid layer_id!!"
        try:
            layer = self._model.layers[layer_id]
        except RuntimeError as e :
            raise e
        finally:
            print('---------------------------------------------------')
            print('-- Network contains the following IDs and shapes --')
            print('---------------------------------------------------')
            for i in range(len(self._model.layers)):
                l = self._model.layers[i]
                print(i, l.__class__.__name__,
                      l.input.shape,
                      '-->',
                      l.output.shape)

        ''' Check the debug Tensor, either create a random one or use the debug_input_tensor'''
        debug_input_tensor = self._debug_create_tensor(layer.input.shape, debug_input_tensor, batch_size)

        self._debug[self.DEBUG.layer] = True
        #self._debug[self.DEBUG.tensor] = debug_input_tensor

        # call function for debugging
        getattr(self, str(layer.__class__.__name__))(debug_input_tensor, layer)

    def debug_model(self, debug_input_tensor=None, batch_size=None, debug_plot=False):
        """
        Function for debugging the entire model using either an own tensor or it is created a random tensor automatically.
        :param debug_input_tensor: Own Tensor for debugging
        :param batch_size: Batch_Size for auto generated tensor. If not specified the batch_size is 1 for higher
        dimensional layer inputs and 10 for one-dimensional inputs
        :param debug_plot: Shows some intermediate plots for the layer actually iterated.
        :return: No return
        """
        debug_input_tensor = self._debug_create_tensor(self._model.input.shape, debug_input_tensor, batch_size)

        ''' Check the debug Tensor, either create a random one or use the debug_input_tensor'''
        self._debug[self.DEBUG.model] = True
        self._debug[self.DEBUG.tensor] = debug_input_tensor
        self._debug[self.DEBUG.plot] = debug_plot

        self.init_logger()
        self.create_EP_Model()

    def _debug_create_tensor(self, input_shape, debug_input_tensor=None, batch_size=None):
        """
        Creates an automatic random tensor of correct shape if debug_input_tensor is None. Otherwise debug_input_tensor
        will be proofed if it has the correct shape.
        :param input_shape: Required input_shape of the layer processing
        :param debug_input_tensor: Own Tensor for debugging
        :param batch_size: Batch_Size for auto generated tensor. If not specified the batch_size is 1 for higher
        dimensional layer inputs and 10 for one-dimensional inputs
        :return: Debug Tensor
        """
        if debug_input_tensor is not None:
            if input_shape[1:] != debug_input_tensor.shape[1:]:
                raise Exception(f'Input_Tensor has incorrect shape {debug_input_tensor.shape}\n'
                                f'Please specify an input tensor with correct shape {input_shape}!')

        if debug_input_tensor is None:
            if batch_size is None:
                batch_size = 1
                if len(input_shape) == 2 and input_shape[1] == 1:
                    batch_size = 10

            input_shape = list(input_shape)
            input_shape[0] = batch_size
            debug_tensor = tf.convert_to_tensor(np.random.choice(np.arange(-10., 10.), input_shape),
                                                dtype=tf.float32)
        else:
            debug_tensor = tf.convert_to_tensor(debug_input_tensor, dtype=tf.float32)

        return debug_tensor

    def _debug_output(self, output):

        ''' If you want to debug the values Eager Execution needs to be enabled'''
        if not tf.executing_eagerly():
            logging.warning('If eager execution is disabled not all debug values can be visualized!'
                            ' Enable Eager Execution for proper debugging!')
        
        # For debugging the layer we want a tensor only
        if self._debug[self.DEBUG.layer]:
            out = self._debug_create_tensor(output.shape)

        elif self._debug[self.DEBUG.model]:
            functor = K.function([self._model.input,
                                  K.symbolic_learning_phase()],
                                 output)  # evaluation function
    
            out = functor([self._debug[self.DEBUG.tensor], False])
            out = tf.convert_to_tensor(out, dtype=tf.float32)

        return out

    def get_layer_evaluation_generator(self, debug_input_tensor=None):

        """
        Todo
        :param debug_input_tensor:
        :return:
        """

        ''' All outputs we wanna have during the evaluation
            1. Input to the layers
            2. Layer output without activation
            3. Layer output with activation
        '''
        outputs = [[layer.input,  # Input to the layers
                    layer.output.op.inputs[0].op.inputs[0],  # Output without Activation
                    layer.output  # Output with activation
                    ] for layer in self._model.layers if layer.__class__.__name__ != 'InputLayer']

        ''' Keras function for getting the layer ouputs'''
        layer_outputs = K.function([self._model.input,
                                    K.symbolic_learning_phase()],
                                   outputs)  # evaluation function

        # predictions = K.function([self._model.layers.input,
        #                           K.symbolic_learning_phase()],
        #                           self._model.layers.output)

        # Create all the intemediate outputs with dropout disables. If Dropout is enables, we have MC-Dropout
        layer_outs = layer_outputs([debug_input_tensor, False])
        # preds = predictions([X_test, False])

        debug_input_tensor = self._debug_create_tensor(self._model.input.shape, debug_input_tensor)

        for i, o in enumerate(zip(layer_outs, self._model.layers[1:])):
            i += 1
            inputs = o[0][0]
            output_b_a = o[0][1]  # Output before activation
            output = o[0][2]  # Output after activation
            l = o[1]
            model_dict = {'layer_name': l.__class__.__name__,
                          'layer': l,
                          'inputs': inputs,
                          'output_b_a': output_b_a,
                          'output': output
                          }
            yield model_dict

    @property
    def model(self):
        return self._ep_model

    @property
    def use_Gaussian_ReLU(self):
        return self._use_Gaussian_ReLU

    def MaxPooling2D(self, error, layer):
        """
            Hint: If you want to debug and watch the output clearly tanspose variables
            e.g.
                inputs_np = inputs.numpy()[0, :, :, ].transpose(2, 0, 1)
                y_np = y.numpy()[0, :, :, ].transpose(2, 0, 1)
                gradient_np = gradient.numpy()[0, :, :, ].transpose(2, 0, 1)

            ############## AVG ##############

            inputs_np

               / 14.0  11.0  16.0  13.0 /
              / 16.0  19.0  16.0  2.0  /
             / 2.0   6.0   11.0  17.0 /
            / 16.0  7.0   19.0  9.0  /

            y_np

             / 19.0  16.0 /
            / 16.0  19.0 /

            gradient_np

               / 0.0   0.0   1.0   0.0  /
              / 0.0   1.0   0.0   0.0  /
             / 0.0   0.0   0.0   0.0  /
            / 1.0   0.0   1.0   0.0  /

            The formula for propagating the error through max-pooling can be described as follows:

            1. Calculate the gradient w.r.t the inputs
            2. The gradient contain 1. for the max values of the input_tensor pooling-window
            3. Calculate the indices of the max values by
                3.1 Flatten the gradients
                3.2 Find all indices of the flatten gradient where the element equals 1.
                    e.g. flatten_gradient = 1 0 0 1 1 0 1 --> idx = 0 3 4 6
            4. Gather all elements from the error based on the indices
            5. Reshape the error


            :param error: Propagated Error Tensor from previous layer --> l-1
            :param layer: Current Layer for doing error Propagation --> l
            :return: Propagated Error for next layer --> l
        """

        if error is None:
            return None

        with tf.GradientTape() as gt:
            ''' By default, GradientTape doesn’t track constants,
             so we must instruct it to with: gt.watch(variable)
             '''
            inputs = layer.input

            # check if one of the debug options is enabled
            if not all((not value or value is None) for value in self._debug.values()):
                inputs = self._debug_output(inputs)

            gt.watch(inputs)
            y = tf.nn.max_pool2d(input=inputs, ksize=layer.pool_size, strides=layer.strides,
                                 padding=layer.padding.upper())

            # calculate the gradient of the pooling output w.r.t the inputs
            gradient = gt.gradient(y, inputs)

        gradient_flatten = tf.reshape(gradient, [-1])
        # use tf.where(tf.equal(gradient_flatten, 1)) instead of tf.where(gradient_flatten == 1)
        # otherwise tf.where will return 0 when eager execution is disabled --> :-(
        idx = tf.reshape(tf.where(tf.equal(gradient_flatten, 1)), [-1])
        error_flatten = tf.reshape(error, [-1])
        error = tf.gather(error_flatten, idx)

        # Reshape to Down-sampled shape
        error = tf.reshape(error, tf.shape(y))

        logging.info(f'Propagate Error {error.shape}')

        return error

    def AveragePooling2D(self, error, layer):
        """
            Hint: If you want to debug and watch the output clearly tanspose variables
            e.g.
            error_np = error.numpy()[0,:,:,].transpose(2,0,1)
            y_np = y.numpy()[0,:,:,].transpose(2,0,1)

            ############## AVG ##############

            error_np
            / 14.0  11.0  16.0  13.0 /
            / 16.0  19.0  16.0  2.0  /
            / 2.0   6.0   11.0  17.0 /
            / 16.0  7.0   19.0  9.0  /

            y_np
            / 15.0  11.7 /
            / 7.75  14.0 /

            gradient = 1/m (m = product of pooling size)
            / 0.25  0.25  0.25  0.25 /
            / 0.25  0.25  0.25  0.25 /
            / 0.25  0.25  0.25  0.25 /
            / 0.25  0.25  0.25  0.25 /

            The formula for propagating the error through average-pooling can be described as follows:

            1. Because the gradient is constant (see above) we can pool (down-sample) the error directly
            2. Calculate the constant gradient w.r.t. the pooling sizes = 1/m (m=product of pooling sizes)
            3. Multiply the squared gradient with the pooled_error

            error_pooled = down-sampling of error
            gradient = 1/m (m=product of pooling sizes)
            error = gradient^2 x error_pooled


            :param error: Propagated Error Tensor from previous layer --> l-1
            :param layer: Current Layer for doing error Propagation --> l
            :return: Propagated Error for next layer --> l
        """

        if error is None:
            return None


        ''' Down-sample the error '''
        error_pooled = tf.nn.avg_pool2d(input=error, ksize=layer.pool_size, strides=layer.strides,
                              padding=layer.padding.upper())

        ''' Calculate the gradient '''
        gradient = tf.math.divide(1., (layer.pool_size[0] * layer.pool_size[1]))

        ''' Propagate the error by multiply the square of the gradient with the pooling output'''
        error = tf.multiply((gradient ** 2), error_pooled)

        logging.info(f'Propagate Error {error.shape}')
        return error

    def Flatten(self, error, layer):
        """
        Flatten the Error based on the layer output shape
        :param error: Propagated Error Tensor from previous layer --> l-1
        :param layer: Current Layer for doing error Propagation --> l
        :return: Flatted Error --> l
        """
        if error is None:
            return None

        logging.info(f'Propagate Error {error.shape}')

        error_flatten_dim = layer.output_shape[1]

        '''Propagate error through Flatten layer'''
        return tf.reshape(tensor=error, shape=(-1, error_flatten_dim))

    def Activation_Eagerly(self, layer_input, error, func):
        """
        Todo
        :param layer_input:
        :param error:
        :param func:
        :return:
        """
        with tf.GradientTape() as gt:
            '''By default, GradientTape doesn’t track constants,
             so we must instruct it to with: gt.watch(variable)
             Todo: Explain how it works
             '''
            gt.watch(layer_input)
            y = func(layer_input)

        gradient = gt.gradient(y, layer_input)
        if func.__name__ != 'softmax':
            return tf.multiply(gradient, error)
        else:
            ''' Calculate the Jacobian of the Softmax activation function.
                Unfortunately it is not possible to use gradienttape to calulate the jacobian
                if eager execution is enabled
            '''
            return self.softmax_jacobian(layer_input=layer_input, error=error)

    @tf.function
    def Activation_not_Eagerly(self, layer_input, error, func):
        """
        Todo
        :param layer_input:
        :param error:
        :param func:
        :return:
        """
        with tf.GradientTape() as gt:
            '''By default, GradientTape doesn’t track constants,
             so we must instruct it to with: gt.watch(variable)
             '''
            gt.watch(layer_input)
            y = func(layer_input)

        if func.__name__ != 'softmax':
            # We need the gradient of the function
            # If we sum by the colums( axis 1) we get the gradient
            # gradient = tf.reduce_sum(jacobian, axis=1)
            gradient = gt.gradient(y, layer_input)
            return tf.multiply(gradient, error)
        else:
            # We need the jacobian of the function
            jacobian = gt.batch_jacobian(y, layer_input)
            return tf.einsum('ijk,ik->ij', jacobian ** 2, error)

    def Activation(self, error, layer):
        """ Activation could be an own layer or part of another layer, thus
            we have to distinguish which input shall be taken.
            For propagating the error we always need the input from affine layers without activation.

            :param error: Propagated Error Tensor from previous layer --> l-1
            :param layer: Current Layer for doing error Propagation --> l
            :return: Propagated Error for next layer --> l
        """

        if error is None:
            return None

        # if the name of the layer is not Activation, we use the output of the previous layer
        # input = w*x
        if layer.__class__.__name__ == "Activation":
            layer_input = layer.input
        elif layer.activation is not tfk.activations.linear:
            layer_input = layer.output.op.inputs[0].op.inputs[0]
        else:
            # Return the error in cases of linear activation (makes no sense to do further processing)
            return error

        # check if one of the debug options is enabled
        if not all((not value or value is None) for value in self._debug.values()):
            layer_input = self._debug_output(layer_input)

        ''' Either do approximation or calculate analytically
            But this is only possible if activation == ReLU
        '''
        if self._use_Gaussian_ReLU and layer.activation.__name__ == 'relu':
            # Do analytical calculation
            error = self.gaussian_relu(layer_input, error)
            logging.info(
                f'Propagate Error through {layer.activation.__name__} --> Analytical (Gaussian ReLU) {error.shape}')
        else:

            ''' If eager execution is disabled it is possible to calculate the Jacobian.
                If eager exectuion is enabled, we only can calculate the Gradient. In case of the Softmax function
                it is not possible to calulate the jacobian in an tensorflow way. I have one implementation but this is
                pretty slow. Thus I created the Jacobian for the softmax activation directly.
                https://math.stackexchange.com/questions/1519367/difference-between-gradient-and-jacobian
            '''
            if tf.executing_eagerly() == False:
                error = self.Activation_not_Eagerly(layer_input, error, layer.activation)
            else:
                error = self.Activation_Eagerly(layer_input, error, layer.activation)

            logging.info(f'Propagate Error through {layer.activation.__name__} --> Approximation {error.shape}')

        if self._debug[self.DEBUG.plot]:
            if not isinstance(error, (np.ndarray)):
                error = error.numpy()
            sns.kdeplot(error.flatten(), cumulative=False, bw=0.001, shade=True,
                        label=f"Error-{layer.__class__.__name__} - {layer.activation.__name__}")
            plt.legend()
            plt.show()
        return error

    def Dense(self, error, layer):
        """
        Todo: Affine Layer
        :param error: Propagated Error Tensor from previous layer --> l-1
        :param layer: Current Layer for doing error Propagation --> l
        :return: Propagated Error for next layer --> l
        """
        if error is None:
            return None

        logging.info(f'Propagate Error {error.shape}')

        # Calculate the variance of linear combination from y=w*x+b --> error = w^2*error
        '''Propagate error through Dense layer'''
        new_error = tf.tensordot(error, layer.weights[0]**2, axes=[[1], [0]])

        if self._debug[self.DEBUG.plot]:
            if not isinstance(error, (np.ndarray)):
                error = error.numpy()
            sns.kdeplot(error.flatten(), cumulative=False, bw=0.001, shade=True,
                        label=f"Error-{layer.__class__.__name__}")
            plt.legend()
            plt.show()

        return self.Activation(new_error, layer)

    def Conv2D(self, error, layer):
        """
            Todo: Propagate error through Conv2D layer
            :param error: Propagated Error Tensor from previous layer --> l-1
            :param layer: Current Layer for doing error Propagation --> l
            :return: Propagated Error for next layer --> l
        """


        if error is None:
            return None

        logging.info(f'Propagate Error {error.shape}')

        error = tf.nn.conv2d(input=error,
                                 filters=layer.weights[0]**2,
                                 strides=layer.strides,
                                 # Hier muss man das groß schreiben im model klein (Conv(Layer) fkt. _get_padding_op)!
                                 padding=layer.padding.upper(),
                                 # conv_utils.py convert_data_format convertiert zu NH...
                                 data_format=convert_data_format(layer.data_format, error.shape.ndims),
                                 dilations=layer.dilation_rate
                                 )
        if self._debug[self.DEBUG.plot]:
            if not isinstance(error, (np.ndarray)):
                error = error.numpy()

            sns.kdeplot(error.flatten(), cumulative=False, bw=0.001, shade=True,
                        label=f"Error-{layer.__class__.__name__}")
            plt.legend()

        ''' Check if an activation function is placed within the layer'''
        return self.Activation(error, layer)

    def BatchNormalization(self, error, layer):
        """
        Todo: Batch_Normalization
        :param error: Propagated Error Tensor from previous layer --> l-1
        :param layer: Current Layer for doing error Propagation --> l
        :return: Propagated Error for next layer --> l
        """
        # if error is None: return None
        logging.info(f'Propagate Error {error.shape}')
        pass

    def Dropout(self, error, layer):
        """
        Calculation of the Variance (Error) of the product of two independent Bernoulli distributed
        random variables based on Leo Goodmans work: "On the Exact Variance of Products (http://www.cs.cmu.edu/~cga/var/2281592.pdf).
        Todo: http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf (10,
        Todo: https://nlp.stanford.edu/pubs/sidaw13fast.pdf

        :param error: Propagated Error Tensor from previous layer --> l-1
        :param layer: Current Layer for doing error Propagation --> l
        :return: New Error
        """
        p = layer.rate  # Dropout rate

        mu_x = layer.input  # Input from Dense to Dropout with ReLU

        # check if one of the debug options is enabled
        if not all((not value or value is None) for value in self._debug.values()):
            mu_x = self._debug_output(mu_x)

        # see https://towardsdatascience.com/simplified-math-behind-dropout-in-deep-learning-6d50f3f47275
        sigma_x = 0  # The self.error we propagate. Sigma_x is unknown in first Dropout layer.
        # set to prior variance of input or zero.
        mu_y = 1 - p  # New Bernoulli mean
        sigma_y = p * (1 - p)  # New Bernoulli variance

        if error is not None:
            sigma_x = error  # Set the self.error to sigma_x the second time we found a Dropout layer in the network

        ''' Calculate the new self.error (variance) (http://www.odelama.com/data-analysis/Commonly-Used-Math-Formulas/)'''
        error = ((mu_x ** 2 * sigma_y) + (mu_y ** 2 * sigma_x) + (sigma_x * sigma_y))

        logging.info(f'Create Error {error.shape}')
        if self._debug[self.DEBUG.plot]:
            if not isinstance(error, (np.ndarray)):
                error = error.numpy()

            sns.kdeplot(error.flatten(), cumulative=False, bw=0.001, shade=True,
                        label=f"Error-{layer.__class__.__name__} creation")
            plt.legend()
            plt.show()
        return error


    def InputLayer(self, error, layer):
        ''' Just a dummy'''
        pass

