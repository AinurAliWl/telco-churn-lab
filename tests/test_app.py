"""Streamlit's app harness checks widget flows without an external browser."""

from streamlit.testing.v1 import AppTest

from churn_lab.data import ROOT


def control(elements, label):
    return next(item for item in elements if item.label == label)


def test_app_starts_and_zero_capacity_produces_no_contacts():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert control(app.metric, "Customers to contact").value == "141"
    control(app.slider, "Contact capacity (%)").set_value(0).run()
    assert not app.exception
    assert control(app.metric, "Customers to contact").value == "0"
    assert control(app.metric, "Expected net value").value == "$0"


def test_model_and_customer_selection_update_without_errors():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    control(app.selectbox, "Inspect a model").set_value("logistic_regression").run()
    assert not app.exception
    assert control(app.metric, "Average precision").value == "0.632"
    select = control(app.selectbox, "Customer ID · ordered by CatBoost risk")
    select.select_index(5).run()
    assert not app.exception


def test_zero_rescue_assumption_filters_unprofitable_contacts():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    control(app.selectbox, "Prioritization").set_value("Highest assumed net value").run()
    control(app.slider, "Assumed rescue rate (%)").set_value(0).run()
    assert not app.exception
    assert control(app.metric, "Customers to contact").value == "0"
