from app.data import load_data
from app.model import build_model
from app.preprocess import split_data


def test_model_fit_and_predict():
    X, y = load_data()
    X_small = X.sample(n=500, random_state=0)
    y_small = y.loc[X_small.index]
    X_train, X_test, y_train, y_test = split_data(X_small, y_small, test_size=0.2, random_state=0)

    model = build_model(n_estimators=50)  # keep test light-weight
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    assert len(preds) == len(y_test)
    assert preds.shape[0] > 0
