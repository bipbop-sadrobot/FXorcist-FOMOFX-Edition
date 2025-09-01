import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import shap
from forex_ai_dashboard.utils.explainability import ModelExplainer

class TestModelExplainer(unittest.TestCase):
    def setUp(self):
        # Create mock model with predict and predict_proba methods
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = np.random.rand(10)
        self.mock_model.predict_proba.return_value = np.random.rand(10, 2)
        
        # Test data
        self.feature_names = ['feature1', 'feature2', 'feature3']
        self.X_test = pd.DataFrame(np.random.rand(10, 3), columns=self.feature_names)
        
    @patch('shap.Explainer')
    def test_init_with_predict_proba(self, mock_shap):
        """Test explainer initialization with predict_proba"""
        explainer = ModelExplainer(self.mock_model, self.feature_names)
        self.assertTrue(mock_shap.called)
        self.assertEqual(explainer.feature_names, self.feature_names)

    @patch('shap.Explainer')
    def test_explain_global(self, mock_shap):
        """Test global explanation generation"""
        # Create mock SHAP Explanation object
        mock_explanation = MagicMock()
        mock_explanation.values = np.random.rand(10, 3)
        mock_explanation.base_values = np.zeros(10)
        mock_explanation.data = self.X_test.values
        mock_shap.return_value.return_value = mock_explanation
        
        explainer = ModelExplainer(self.mock_model, self.feature_names)
        result = explainer.explain(self.X_test)
        
        self.assertIn('shap_values', result)
        self.assertIn('feature_importance', result)
        self.assertEqual(len(result['feature_importance']), 3)  # 3 features
        
    @patch('shap.Explainer')
    def test_explain_instance(self, mock_shap):
        """Test instance explanation"""
        # Create mock SHAP Explanation object
        mock_explanation = MagicMock()
        mock_explanation.values = np.random.rand(1, 3)
        mock_explanation.base_values = np.zeros(1)
        mock_explanation.data = self.X_test.iloc[[0]].values
        mock_shap.return_value.return_value = mock_explanation
        
        explainer = ModelExplainer(self.mock_model, self.feature_names)
        result = explainer.explain_instance(self.X_test.iloc[[0]])
        
        self.assertIn('shap_values', result)
        self.assertIn('force_plot', result)

    @patch('shap.Explainer')
    def test_feature_importance_ordering(self, mock_shap):
        """Test feature importance ranking"""
        # Create mock SHAP Explanation object with known values
        mock_explanation = MagicMock()
        mock_explanation.values = np.array([
            [0.1, 0.5, 0.2],  # Feature2 should be most important
            [0.3, 0.4, 0.1]
        ])
        mock_explanation.base_values = np.zeros(2)
        mock_explanation.data = self.X_test.iloc[:2].values
        mock_shap.return_value.return_value = mock_explanation
        
        explainer = ModelExplainer(self.mock_model, self.feature_names)
        result = explainer.explain(self.X_test.iloc[:2])
        
        # Check feature importance ordering
        self.assertEqual(result['feature_importance'].iloc[0]['feature'], 'feature2')
        self.assertEqual(result['feature_importance'].iloc[1]['feature'], 'feature1')

if __name__ == '__main__':
    unittest.main()
