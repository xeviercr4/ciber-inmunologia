"""
models.py — Los tres modelos del proyecto.

  - MLP:             red densa sencilla, usada como línea base.
  - FlowTransformer: trata cada característica del flujo como un token y usa
                     atención para relacionarlas entre sí.
  - Autoencoder:     aprende a reconstruir tráfico normal; lo que reconstruye
                     mal se considera anómalo.

Los dos clasificadores (MLP y FlowTransformer) tienen la misma forma:
    modelo.encoder -> convierte el flujo en un vector de tamaño D_MODELO
    modelo.cabeza  -> convierte ese vector en una puntuación por familia
Tenerlos separados permite federar solo el encoder y dejar la cabeza local
en cada nodo (la variante personalizada).
"""
import torch
from torch import nn

from src.config import D_MODELO, FAMILIAS, NUM_CABEZAS, NUM_CAPAS


class MLP(nn.Module):
    """Línea base: dos capas ocultas + capa de clasificación."""

    def __init__(self, n_caracteristicas: int, n_clases: int = len(FAMILIAS)):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_caracteristicas, 128), nn.ReLU(),
            nn.Linear(128, D_MODELO), nn.ReLU(),
        )
        self.cabeza = nn.Linear(D_MODELO, n_clases)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cabeza(self.encoder(x))


class EncoderFlujo(nn.Module):
    """
    Encoder del FlowTransformer.

    1. Tokenización: cada característica (un número) se convierte en un vector
       de tamaño d:   token_i = valor_i * peso_i + sesgo_i
       Cada característica tiene su propio peso y sesgo, así el modelo sabe
       "cuál" característica es cada token (hace las veces de posición).
    2. Se agrega al principio un token especial [CLS] que resume todo el flujo.
    3. Las capas de atención dejan que cada token "mire" a los demás.
    4. La salida es el vector final del token [CLS].
    """

    def __init__(self, n_caracteristicas: int, d: int, cabezas: int, capas: int):
        super().__init__()
        limite = d ** -0.5
        self.peso = nn.Parameter(torch.empty(n_caracteristicas, d).uniform_(-limite, limite))
        self.sesgo = nn.Parameter(torch.empty(n_caracteristicas, d).uniform_(-limite, limite))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        capa = nn.TransformerEncoderLayer(
            d_model=d, nhead=cabezas, dim_feedforward=2 * d, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(capa, num_layers=capas, enable_nested_tensor=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [lote, n_caracteristicas]
        tokens = x.unsqueeze(-1) * self.peso + self.sesgo        # [lote, n_caracteristicas, d]
        cls = self.cls.expand(len(x), -1, -1)                    # [lote, 1, d]
        salida = self.transformer(torch.cat([cls, tokens], dim=1))
        return salida[:, 0]                                      # vector del token [CLS]: [lote, d]


class FlowTransformer(nn.Module):
    """Encoder con atención multi-cabeza + cabeza de clasificación separada."""

    def __init__(self, n_caracteristicas: int, n_clases: int = len(FAMILIAS)):
        super().__init__()
        self.encoder = EncoderFlujo(n_caracteristicas, D_MODELO, NUM_CABEZAS, NUM_CAPAS)
        self.cabeza = nn.Linear(D_MODELO, n_clases)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cabeza(self.encoder(x))


class Autoencoder(nn.Module):
    """
    Comprime el flujo a 8 números y lo vuelve a expandir. Se entrena solo con
    tráfico benigno: aprende a reconstruir bien lo normal y mal lo extraño.
    """

    def __init__(self, n_caracteristicas: int):
        super().__init__()
        self.codificador = nn.Sequential(nn.Linear(n_caracteristicas, 32), nn.ReLU(), nn.Linear(32, 8), nn.ReLU())
        self.decodificador = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, n_caracteristicas))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decodificador(self.codificador(x))

    def error(self, x: torch.Tensor) -> torch.Tensor:
        """Error de reconstrucción de cada flujo (más alto = más anómalo)."""
        return ((self(x) - x) ** 2).mean(dim=1)
