import json
import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")


# Базовые классы нейросетей
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Neuron:
    def __init__(self, weights, bias):
        self.weights = np.array(weights)
        self.bias = bias

    def feedforward(self, inputs):
        return sigmoid(np.dot(inputs, self.weights) + self.bias)


class LinearNeuron:
    def __init__(self, weights, bias):
        self.weights = np.array(weights)
        self.bias = bias

    def feedforward(self, inputs):
        return np.dot(inputs, self.weights) + self.bias


class LoadedNetwork:
    def __init__(self, hidden_params, output_params, is_linear=False):
        self.hidden_layer = [Neuron(p["weights"], p["bias"]) for p in hidden_params]
        if is_linear:
            self.output_layer = [LinearNeuron(p["weights"], p["bias"]) for p in output_params]
        else:
            self.output_layer = [Neuron(p["weights"], p["bias"]) for p in output_params]

    def predict(self, inputs):
        hidden_outs = np.array([n.feedforward(inputs) for n in self.hidden_layer])
        outputs = np.array([n.feedforward(hidden_outs) for n in self.output_layer])
        return outputs


# Загружаем модель классификации
with open("classification_weights.json", "r", encoding="utf-8") as f:
    c_data = json.load(f)
net_class = LoadedNetwork(c_data["hidden"], c_data["output"], is_linear=False)
mn_c, mx_c, names_c = np.array(c_data["mn"]), np.array(c_data["mx"]), c_data["names"]

# Загружаем модель регрессии
with open("regression_weights.json", "r", encoding="utf-8") as f:
    r_data = json.load(f)
net_reg = LoadedNetwork(r_data["hidden"], r_data["output"], is_linear=True)
mn_r, mx_r = np.array(r_data["mn_x"]), np.array(r_data["mx_x"])
mn_y, mx_y = r_data["mn_y"], r_data["mx_y"]


@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.post("/classify")
def classify(
        request: Request,
        f1: float = Form(...),
        f2: float = Form(...),
        f3: float = Form(...),
        f4: float = Form(...)
):
    features = np.array([f1, f2, f3, f4])
    normalized = (features - mn_c) / (mx_c - mn_c)
    preds = net_class.predict(normalized)
    class_idx = int(np.argmax(preds))
    result = names_c[class_idx]

    return templates.TemplateResponse(request=request, name="index.html", context={
        "class_result": f"Класс вина: {result} (вероятности: {np.round(preds, 3).tolist()})"
    })


@app.post("/regress")
def regress(
        request: Request,
        r1: float = Form(...),
        r2: float = Form(...),
        r3: float = Form(...)
):
    features = np.array([r1, r2, r3])
    normalized = (features - mn_r) / (mx_r - mn_r)
    pred_norm = net_reg.predict(normalized)[0]
    # Денормализация ответа
    prediction = pred_norm * (mx_y - mn_y) + mn_y

    return templates.TemplateResponse(request=request, name="index.html", context={
        "reg_result": f"Предсказанное значение (4-й признак): {round(float(prediction), 4)}"
    })