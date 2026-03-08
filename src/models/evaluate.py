import shap

explainer = shap.Explainer(best_model, X_train)

shap_values = explainer(X_test)

shap.plots.bar(shap_values)