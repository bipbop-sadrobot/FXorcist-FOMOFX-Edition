import logging
from overfitting_manager import OverfittingManager

class OverfittingSnatcher:
    """
    Catches overfitting warnings and proactively manages interventions.
    Works in conjunction with OverfittingManager, not standalone.
    """

    def __init__(self):
        self.manager = OverfittingManager(patience=5, cooldown=2)
        self.intervention_effectiveness = {}  # Track what works
        logging.basicConfig(level=logging.INFO, format="[OverfittingSnatcher] %(message)s")

    def monitor_epoch(self, epoch, train_loss, val_loss):
        """
        Called at the end of each epoch. Handles interventions automatically.

        Args:
            epoch (int): Current epoch
            train_loss (float): Latest training loss
            val_loss (float): Latest validation loss
        """
        result = self.manager.check_overfitting(train_loss, val_loss)

        # Log warning
        logging.info(f"Epoch {epoch}: {result['warning']}")

        if result["actions"]:
            self._execute_actions(result["actions"], epoch)

        # Store metadata (could be fed back into training AI)
        return result

    def _execute_actions(self, actions, epoch):
        """
        Executes or redirects interventions.
        Here we only log, but hooks are in place for actual changes.
        """
        for action in actions:
            logging.info(f"Epoch {epoch}: Executing intervention -> {action}")

            # Example redirect hooks (AI can map these to real ops)
            if "learning rate" in action.lower():
                self._redirect("Adjust learning rate scheduler")
            elif "dropout" in action.lower():
                self._redirect("Update model regularization")
            elif "augment" in action.lower():
                self._redirect("Activate data augmentation pipeline")
            elif "checkpoint" in action.lower():
                self._redirect("Restore best checkpoint")
            elif "batch size" in action.lower():
                self._redirect("Reconfigure dataloader")

            # Track usage frequency
            self.intervention_effectiveness[action] = self.intervention_effectiveness.get(action, 0) + 1

    def _redirect(self, instruction):
        """
        Intuitive redirection stub.
        Replace this with calls into your training controller.
        """
        logging.info(f" → Redirected: {instruction}")

    def get_intervention_summary(self):
        """
        Provides a summary of what was tried most often.
        Useful for the AI to refine strategy over time.
        """
        return dict(sorted(self.intervention_effectiveness.items(), key=lambda x: -x[1]))