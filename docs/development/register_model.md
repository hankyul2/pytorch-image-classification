# Register Model Summary

In this post, I will summarize what I have done for developing `register_model` function. 
`register_model` has 2 main functions.

1. `register_model` collect parameter type and default value from its signature.
    To collect parameter type and default value, you should use `inspect.signature` that returns
    information about parameter name, parameter default value, parameter data type.
    By using `inspect.signature`, argument lists for each model is created, which can be used for 
    making subparsers of main argparser.
2. `register_model` also collect each model's configuration and store in single dictionary to
    be used in `create_model`. During making config dict for each model, comparability between
    config and model signature is also checked.