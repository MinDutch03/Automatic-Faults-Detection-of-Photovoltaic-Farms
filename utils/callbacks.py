# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Callback utils
"""
from typing import Any, Callable, Dict, List


class Callbacks:
    """"
    Handles all registered callbacks for YOLOv5 Hooks
    """

    def __init__(self):
        # Define the available callbacks
        self._callbacks = {
            'on_pretrain_routine_start': [],
            'on_pretrain_routine_end': [],
            'on_train_start': [],
            'on_train_epoch_start': [],
            'on_train_batch_start': [],
            'optimizer_step': [],
            'on_before_zero_grad': [],
            'on_train_batch_end': [],
            'on_train_epoch_end': [],
            'on_val_start': [],
            'on_val_batch_start': [],
            'on_val_image_end': [],
            'on_val_batch_end': [],
            'on_val_end': [],
            'on_fit_epoch_end': [],  # fit = train + val
            'on_model_save': [],
            'on_train_end': [],
            'on_params_update': [],
            'teardown': [],
        }
        self.stop_training = False  # set True to interrupt training

    def register_action(self, hook: str, name: str = '', callback: Callable = None) -> None:
        """
        Register a new action to a callback hook

        Args:
            hook: The callback hook name to register the action to
            name: The name of the action for later reference
            callback: The callback to fire
        """
        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"
        assert callable(callback), f"callback '{callback}' is not callable"
        self._callbacks[hook].append({'name': name, 'callback': callback})

    def get_registered_actions(self, hook: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """"
        Returns all the registered actions by callback hook

        Args:
            hook: The name of the hook to check, defaults to all
        """
        return self._callbacks[hook] if hook else self._callbacks

    def run(self, hook: str, *args, **kwargs) -> None:
        """
        Loop through the registered actions and fire all callbacks

        Args:
            hook: The name of the hook to check, defaults to all
            args: Arguments to receive from YOLOv5
            kwargs: Keyword Arguments to receive from YOLOv5
        """
        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"

        for logger in self._callbacks[hook]:
            try:
                # Get the number of parameters the callback accepts
                import inspect
                callback_params = inspect.signature(logger['callback']).parameters
                num_params = len(callback_params)

                # If the callback accepts variable arguments (*args), pass all arguments
                if any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in callback_params.values()):
                    logger['callback'](*args, **kwargs)
                # Otherwise, pass only the number of arguments the callback can accept
                else:
                    logger['callback'](*args[:num_params], **kwargs)
            except Exception as e:
                print(f"Error in callback {logger['name']} for hook {hook}: {str(e)}")
                # Optionally, raise the exception if you want to stop execution
                # raise e
