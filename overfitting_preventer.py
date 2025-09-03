import logging

class OverfittingPreventer:
    """
    Proactively applies proven strategies to reduce overfitting risk.
    Designed to be plugged into a training pipeline as a guardrail system.
    """

    def __init__(self, gap_threshold=0.05, min_epoch=3):
        """
        Args:
            gap_threshold (float): Acceptable train/val gap before preventions trigger
            min_epoch (int): Minimum epochs before triggering preventions
        """
        self.gap_threshold = gap_threshold
        self.min_epoch = min_epoch
        self.active_preventions = []
        self.applied_strategies = {}
        logging.basicConfig(level=logging.INFO, format="[OverfittingPreventer] %(message)s")

    def suggest_preventions(self, epoch, train_loss, val_loss):
        """
        Suggest proactive adjustments based on loss patterns.

        Args:
            epoch (int): Current epoch
            train_loss (float): Training loss
            val_loss (float): Validation loss

        Returns:
            dict: Preventions and reasoning
        """
        gap = abs(val_loss - train_loss)
        preventions = {"epoch": epoch, "preventions": [], "reason": None}

        # Avoid premature triggers
        if epoch < self.min_epoch:
            preventions["reason"] = f"Epoch {epoch} < min_epoch ({self.min_epoch}) → Monitoring only."
            return preventions

        # If gap is stable, no intervention
        if gap < self.gap_threshold:
            preventions["reason"] = f"Train/Val gap {gap:.3f} stable < threshold {self.gap_threshold}."
            return preventions

        preventions["reason"] = (
            f"Train/Val gap {gap:.3f} exceeds threshold {self.gap_threshold} → "
            f"Activating compensatory mechanisms."
        )

        # Industry-standard strategies
        strategies = [
            ("Data Augmentation", "Add or intensify augmentation (noise, flips, crops, mixup, cutmix)."),
            ("Dropout Adjustment", "Increase dropout rate slightly to reduce co-adaptation."),
            ("Regularization", "Apply/strengthen L2 weight decay."),
            ("Learning Rate Scheduling", "Reduce learning rate (factor ~0.8)."),
            ("Batch Normalization", "Ensure normalization layers are active and tuned."),
            ("Early Checkpoint", "Save checkpoint in case of rollback need."),
            ("Smaller Batch Size", "Switch to smaller batch size to regularize."),
        ]

        # Apply strategies in a prioritized sequence
        for name, description in strategies:
            if name not in self.applied_strategies:  # Avoid spamming same fix
                preventions["preventions"].append({name: description})
                self.applied_strategies[name] = epoch
                logging.info(f"Epoch {epoch}: Prevention applied → {name} :: {description}")
                break  # Apply one new prevention per trigger cycle

        self.active_preventions.extend(preventions["preventions"])
        return preventions

    def get_active_preventions(self):
        """Return all unique active preventions so far."""
        return list({list(p.keys())[0] for p in self.active_preventions})

    def get_prevention_history(self):
        """Return dictionary of strategies and first applied epochs."""
        return self.applied_strategies