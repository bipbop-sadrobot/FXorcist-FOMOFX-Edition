import random
import logging

class OverfittingManager:
    """
    A modular overfitting handler that can be plugged into an AI training loop.
    It doesn't stop training but provides proactive strategies to mitigate and
    manage overfitting once detected.
    """

    def __init__(self, patience=5, cooldown=2):
        """
        Args:
            patience (int): Number of epochs to tolerate before forcing stronger interventions.
            cooldown (int): Number of epochs to wait after an intervention before another.
        """
        self.patience = patience
        self.cooldown = cooldown
        self.best_val_loss = float("inf")
        self.bad_epochs = 0
        self.cooldown_counter = 0
        self.intervention_history = []

        logging.basicConfig(level=logging.INFO, format="[OverfittingManager] %(message)s")

    def check_overfitting(self, train_loss, val_loss):
        """
        Main entrypoint. Call this every epoch with training + validation loss.

        Args:
            train_loss (float): Latest training loss
            val_loss (float): Latest validation loss

        Returns:
            dict: Recommended actions and metadata
        """
        action_plan = {"warning": None, "actions": [], "metadata": {}}

        # Improvement check
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.bad_epochs = 0
            action_plan["warning"] = "Validation improved. No intervention."
        else:
            self.bad_epochs += 1
            action_plan["warning"] = f"Potential overfitting detected (bad_epochs={self.bad_epochs})."

        # Cooldown check
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            action_plan["actions"].append("Cooldown active, no major changes.")
            return action_plan

        # Escalating interventions
        if self.bad_epochs >= self.patience:
            action_plan["warning"] = "Overfitting sustained. Intervening."
            interventions = self._suggest_interventions(train_loss, val_loss)
            action_plan["actions"].extend(interventions)
            self.intervention_history.append(interventions)
            self.cooldown_counter = self.cooldown

        action_plan["metadata"] = {
            "best_val_loss": self.best_val_loss,
            "bad_epochs": self.bad_epochs,
            "cooldown_remaining": self.cooldown_counter,
            "intervention_history": self.intervention_history,
        }
        return action_plan

    def _suggest_interventions(self, train_loss, val_loss):
        """
        Suggest interventions to mitigate overfitting.
        Randomizes to avoid repeating the same adjustment endlessly.
        """
        interventions = []

        options = [
            "Reduce learning rate by factor of 0.5",
            "Apply dropout increase (e.g., +0.1)",
            "Enable stronger weight decay",
            "Augment data (rotations, noise, crops, etc.)",
            "Switch to best saved checkpoint",
            "Introduce gradient clipping",
            "Use smaller batch size for regularization effect",
        ]

        # Pick 2-3 interventions at a time
        chosen = random.sample(options, k=min(3, len(options)))
        interventions.extend(chosen)

        return interventions