import tensorflow as tf

#-------------------- FUNCTIONS TO GET INFORMATIONS ----------------------------------------
def get_model_layers_info(model, ignore_last_layer=False):
    """
    Returns list of layers and number of units separated by ";"
    f.e. "relu24;linear1"

    optional parameters:
    ignore_last_layer - if set to True then our output does not contain last layer
    """
    layer_strings = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer_strings.append(layer.activation.__name__ + str(layer.units))

    if ignore_last_layer:
        layer_strings = layer_strings[:-1]
    result_string = ';'.join(layer_strings)
    return result_string

def get_model_total_layers_params_and_units(model):
    """
    Returns: total_params, layers_cnt, total_units
    """
    total_params = model.count_params()
    layers_cnt = len(model.layers)
    total_units = sum([layer.output_shape[-1] for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)])
    return total_params, layers_cnt, total_units

def get_model_description(model, pat="unknown", bs="unknown", comment=""):
    """
    INPUTS:
    model - model object to generate description
    pat - patience - integer value of patience (this is defined with callback and model iteself does not contain that value)
    bs - batch_size - integer value - defined while fitting the model
    comment - any other note that was not covered by this function and was worth keeping

    OUTPUT:
    string taht describes the model hiperparameters, f.e.:
    [relu96;relu12][kernel_reg:None][bias_reg:None][activity_reg:None][opt:Adam][lr:0.001][bs:unknown][pat:unknown][comment:]
    """
    #Get layers list
    list_of_layers = get_model_layers_info(model, ignore_last_layer=True)
        
    #Get info about reguralizers
    kernel_reg = model.layers[0].kernel_regularizer
    bias_reg = model.layers[0].bias_regularizer
    activity_reg = model.layers[0].activity_regularizer

    if kernel_reg is not None:
        kernel_reg_name = kernel_reg.__class__.__name__
        kernel_reg_value = kernel_reg.get_config()[kernel_reg_name.lower()]
        kernel_reg_str = f"{kernel_reg_name}-{round(kernel_reg_value,6)}"
    else:
        kernel_reg_str = "None"

    if bias_reg is not None:
        bias_reg_name = bias_reg.__class__.__name__
        bias_reg_value = bias_reg.get_config()[bias_reg_name.lower()]
        bias_reg_str = f"{bias_reg_name}-{round(bias_reg_value,6)}"
    else:
        bias_reg_str = "None"

    if activity_reg is not None:
        activity_reg_name = activity_reg.__class__.__name__
        activity_reg_value = activity_reg.get_config()[activity_reg_name.lower()]
        activity_reg_str = f"{activity_reg_name}-{round(activity_reg_value,6)}"
    else:
        activity_reg_str = "None"
        
    #Get info about optimizer
    optimizer_name = model.optimizer.get_config()['name']
    lr = str(round(model.optimizer.get_config()['learning_rate'], 6))

    return f"[{list_of_layers}][kernel_reg:{kernel_reg_str}][bias_reg:{bias_reg_str}][activity_reg:{activity_reg_str}][opt:{optimizer_name}][lr:{lr}][bs:{bs}][pat:{pat}][comment:{comment}]"


def print_model_architecture(model):
    """Used for debuging if generate_model_architecture provides desired results"""
    for layer in model.layers:
        if layer.kernel_regularizer is not None:
            kernel_reg_type = type(layer.kernel_regularizer).__name__
            kernel_reg_strength = layer.kernel_regularizer.get_config()[kernel_reg_type.lower()]
            kernel_reg_info = f"{kernel_reg_type}({kernel_reg_strength:.2f})"
        else:
            kernel_reg_info = "None"
        if layer.bias_regularizer is not None:
            bias_reg_type = type(layer.bias_regularizer).__name__
            bias_reg_strength = layer.bias_regularizer.get_config()[bias_reg_type.lower()]
            bias_reg_info = f"{bias_reg_type}({bias_reg_strength:.2f})"
        else:
            bias_reg_info = "None"
        if layer.activity_regularizer is not None:
            activity_reg_type = type(layer.activity_regularizer).__name__
            activity_reg_strength = layer.activity_regularizer.get_config()[activity_reg_type.lower()]
            activity_reg_info = f"{activity_reg_type}({activity_reg_strength:.2f})"
        else:
            activity_reg_info = "None"
        print(
            f"Layer: {layer.activation.__name__}{str(layer.units)},\t"
            f"kernel_reg: {kernel_reg_info},\t"
            f"bias_reg: {bias_reg_info},\t"
            f"activity_reg: {activity_reg_info})"
            )
#--------------------  /FUNCTIONS TO GET INFORMATIONS ----------------------------------------


#--------------------  FUNCTIONS TO CREATE MODELS --------------------------------------------
def generate_model_architecture(hidden_layers_list, kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, input_shape=(50,), output_layer=None):
    """
    Creates model object basing on given input
    INPUTS:
    hidden_layers_list (required)
        - list of strings described above f.e. ['relu96', 'relu12']
    kernel_regularizer (None) - None/regularizer_object/list
        - None - Non kernel regularizer would be used
        - regularizer_object - signle kernel regularizer that would be applied to all hidden layers f.e regularizers.l2(0.01)
    bias_regularizer (None) - None/regularizer_object/list
        - None - Non bias regularizer would be used
        - regularizer_object - signle bias regularizer that would be applied to all hidden layers f.e regularizers.l2(0.01)
    activity_regularizer (None) - None/regularizer_object/list
        - None - Non activity regularizer would be used
        - regularizer_object - signle activity regularizer that would be applied to all hidden layers f.e regularizers.l2(0.01)
    input_shape (50, ) - standard format of input_shape for model
    output_layer (None) - ouput lajer object, None value is replaced by: tf.keras.layers.Dense(1, activation='linear')
    """
    import re
    import tensorflow as tf
    if output_layer is None:
        output_layer = tf.keras.layers.Dense(1, activation='linear') #Pick linear layer as default output

    # Creating empty model
    model = tf.keras.Sequential()

    for index, layer in enumerate(hidden_layers_list):
        #Get variables for each layer
        activation = re.findall(r'[a-z]+', layer)[0]
        units = int(re.findall(r'\d+', layer)[0])

        # input_shape provided to any layer after first one would be ignored by tensorflow automaticaly
        model.add(tf.keras.layers.Dense(units, activation=activation, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, input_shape=input_shape))
    
    #Add output layer
    model.add(output_layer)
    return model
#--------------------  FUNCTIONS TO CREATE MODELS --------------------------------------------
