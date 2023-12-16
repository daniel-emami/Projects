from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from matplotlib import pyplot as py



# ============================================================
# ============================================================
#                 FUNCTIONS FOR trad_ML_META
# ============================================================
# ============================================================

def plot_feature_importance(fit, X_test, y_test):
    if type(fit) == RandomForestClassifier:
        importance = fit.feature_importances_
        model = 'Random Forest'
    elif type(fit) == GradientBoostingClassifier:
        importance = fit.feature_importances_
        model = 'Gradient Boosting'
    elif type(fit) == LogisticRegression:
        importance = fit.coef_
        model = 'Logistic Regression'
    elif type(fit) == svm.SVC:
        importance = fit.coef_
        model = 'SVM'
    else:
        print("Error: fit must be a sklearn model")
        return

    # Reshape importance array
    importance = importance.ravel()

    feature_names = ["period_count", "word_count", "character_count", "sentiment_analysis", "long_words_count", "lix", "ci_words_count"]

    # Summarize feature importance

    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))

    # plot feature importance
    py.figure(figsize=(10, 6))
    py.bar(feature_names, importance, tick_label=feature_names, width=0.8)
    py.xticks(rotation=45)
    py.subplots_adjust(bottom=0.25)
    py.title(f'Feature Importance {model} fit on "Constructive Articles Metadata"')
    # Add accuracy score to plot
    accuracy = fit.score(X_test, y_test)
    py.text(0.5, -0.45, f"{model} - Test Accuracy: {accuracy:.2%}", ha='center', transform=py.gca().transAxes, fontsize=16)
    py.show()