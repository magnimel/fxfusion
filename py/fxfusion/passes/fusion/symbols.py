class Fusion:
    @staticmethod
    def relu(*args) -> None:
        raise RuntimeError("Symbolic Op: relu should only execute in the C++ runtime engine.")

    @staticmethod
    def conv2d(*args) -> None:
        raise RuntimeError("Symbolic Op: conv2d should only execute in the C++ runtime engine.")

    @staticmethod
    def conv2d_relu(*args) -> None:
        raise RuntimeError("Symbolic Op: conv2d_relu should only execute in the C++ runtime engine.")

    @staticmethod
    def linear(*args) -> None:
        raise RuntimeError("Symbolic Op: linear should only execute in the C++ runtime engine.")

    @staticmethod
    def linear_relu(*args) -> None:
        raise RuntimeError("Symbolic Op: linear_relu should only execute in the C++ runtime engine.")

    @staticmethod
    def add_relu(*args) -> None:
        raise RuntimeError("Symbolic Op: add_relu should only execute in the C++ runtime engine.")

    @staticmethod
    def layernorm(*args) -> None:
        raise RuntimeError("Symbolic Op: layernorm should only execute in the C++ runtime engine.")

    @staticmethod
    def add_layernorm(*args) -> None:
        raise RuntimeError("Symbolic Op: add_layernorm should only execute in the C++ runtime engine.")

    @staticmethod
    def embedding(*args) -> None:
        raise RuntimeError("Symbolic Op: embedding should only execute in the C++ runtime engine.")

    @staticmethod
    def qkv_linear(*args) -> None:
        raise RuntimeError("Symbolic Op: qkv_linear should only execute in the C++ runtime engine.")

    @staticmethod
    def attention(*args) -> None:
        raise RuntimeError("Symbolic Op: attention should only execute in the C++ runtime engine.")

    @staticmethod
    def residual_add(*args) -> None:
        raise RuntimeError("Symbolic Op: residual_add should only execute in the C++ runtime engine.")

    @staticmethod
    def mha(*args) -> None:
        raise RuntimeError("Symbolic Op: mha should only execute in the C++ runtime engine.")

    @staticmethod
    def feedforward(*args) -> None:
        raise RuntimeError("Symbolic Op: feedforward should only execute in the C++ runtime engine.")