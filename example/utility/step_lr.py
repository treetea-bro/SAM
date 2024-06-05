class StepLR:
    """total_epochs 기준으로 특정 구간에 진입 할때마다 점진적으로
    learning_rate를 감소시켜 minima에 수렴할 수 있도록 도와주는 클래스"""

    def __init__(self, optimizer, learning_rate: float, total_epochs: int):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.base = learning_rate

    def __call__(self, epoch):
        if epoch < self.total_epochs * 3 / 10:
            lr = self.base
        elif epoch < self.total_epochs * 6 / 10:
            lr = self.base * 0.2
        elif epoch < self.total_epochs * 8 / 10:
            lr = self.base * 0.2**2
        else:
            lr = self.base * 0.2**3

        # 특정구간에서 계산된 learning_rate를
        # 옵티마이저에 실제로 적용.
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]
