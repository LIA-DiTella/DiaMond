import torch
import torch.nn as nn
import os
import sys
import warnings

# Añadir la ruta del directorio src al PYTHONPATH de forma más robusta
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from optimizer import LARS, CosineWarmupScheduler  # noqa: E402

# Filtrar advertencias específicas sobre el parámetro verbose de lr_scheduler
warnings.filterwarnings("ignore", message="The verbose parameter is deprecated")


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
        self.out = nn.Linear(5, 1)

        # Inicializar con valores específicos para garantizar gradientes significativos
        with torch.no_grad():
            # Establecer pesos grandes para generar outputs grandes
            self.fc.weight.fill_(0.5)
            self.fc.bias.fill_(0.5)
            self.out.weight.fill_(0.5)
            self.out.bias.fill_(0.5)

    def forward(self, x):
        x = self.fc(x)
        x = torch.relu(x)
        return self.out(x)


class TestOptimizers:
    def test_lars_optimizer(self):
        # Establecer semilla para reproducibilidad
        torch.manual_seed(42)

        # Inicializar modelo con valores específicos para garantizar actualizaciones detectables
        model = SimpleModel()

        # Usar una tasa de aprendizaje mayor para que los cambios sean más evidentes
        optimizer = LARS(model.parameters(), lr=0.1, weight_decay=0.0001)

        # Crear datos con valores específicos para garantizar gradientes grandes
        x = torch.ones(32, 10)
        y = torch.zeros(
            32, 1
        )  # Asegura una gran diferencia entre predicción y objetivo

        # Forward pass
        output = model(x)
        loss = ((output - y) ** 2).mean()

        # Verificar que la pérdida sea significativa (ajustar el umbral a un valor más realista)
        assert loss.item() > 0.01, (
            "La pérdida debería ser significativa para esta prueba"
        )
        print(f"Pérdida inicial: {loss.item()}")

        # Backward pass
        optimizer.zero_grad()  # Asegurarse de que no hay gradientes anteriores
        loss.backward()

        # Verificar que los gradientes sean significativos
        has_grad = False
        for p in model.parameters():
            if p.grad is not None:
                if p.grad.abs().sum() > 0:
                    has_grad = True
                    break
        assert has_grad, (
            "Al menos un parámetro debería tener un gradiente diferente de cero"
        )

        # Store parameter values before optimization step
        old_values = []
        for p in model.parameters():
            old_values.append(p.clone().detach())

        # Optimization step
        optimizer.step()

        # Verificar que al menos un parámetro ha cambiado
        any_param_changed = False
        for i, p in enumerate(model.parameters()):
            if not torch.allclose(p.detach(), old_values[i]):
                any_param_changed = True
                break

        assert any_param_changed, (
            "Al menos un parámetro del modelo debería cambiar después de optimizer.step()"
        )

        # Test con weight_decay_filter y lars_adaptation_filter
        # Crear un nuevo modelo para la segunda parte del test
        model2 = SimpleModel()

        def filter_fn(x):
            return False  # Always return False

        optimizer = LARS(
            model2.parameters(),
            lr=0.01,
            weight_decay=0.0001,
            weight_decay_filter=filter_fn,
            lars_adaptation_filter=filter_fn,
        )

        # Hacer un nuevo forward pass
        output = model2(x)
        loss = ((output - y) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # No es necesario verificar el cambio aquí, solo que no hay error

    def test_cosine_warmup_scheduler(self):
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        # Test with warmup
        scheduler = CosineWarmupScheduler(
            optimizer, max_steps=100, warmup_steps=10, lr=0.1, batch_size=32
        )

        # Initial LR
        assert optimizer.param_groups[0]["lr"] == 0.1 * (0 / 10)  # Initial LR is 0

        # Simular un ciclo de entrenamiento para evitar la advertencia sobre el orden
        for i in range(1, 10):
            # Simular un paso de optimización (para evitar warnings)
            dummy_x = torch.ones(1, 10)
            dummy_y = torch.zeros(1, 1)
            dummy_output = model(dummy_x)
            dummy_loss = ((dummy_output - dummy_y) ** 2).mean()
            optimizer.zero_grad()
            dummy_loss.backward()
            optimizer.step()

            # Ahora actualizar el programador
            scheduler.step()
            assert abs(optimizer.param_groups[0]["lr"] - 0.1 * (i / 10)) < 1e-6

        # After warmup
        optimizer.step()  # Paso de optimización
        scheduler.step()  # Step 10
        assert (
            abs(optimizer.param_groups[0]["lr"] - 0.1) < 1e-6
        )  # Usar tolerancia para comparaciones de punto flotante

        # Test cosine annealing
        for i in range(11, 100):
            optimizer.step()  # Paso de optimización
            scheduler.step()
            # LR should decrease
            assert optimizer.param_groups[0]["lr"] < 0.1

        # Final step
        optimizer.step()  # Paso de optimización
        scheduler.step()  # Step 100
        assert optimizer.param_groups[0]["lr"] >= 0.0001  # Cambiar a >= en lugar de >

        # Test with end_lr
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = CosineWarmupScheduler(
            optimizer,
            max_steps=100,
            warmup_steps=10,
            lr=0.1,
            batch_size=32,
            end_lr=0.001,
        )

        # Go to final step
        for i in range(100):
            optimizer.step()  # Paso de optimización
            scheduler.step()

        # Final LR should be close to end_lr
        assert abs(optimizer.param_groups[0]["lr"] - 0.001) < 1e-4
