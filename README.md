# Deepsekaggler - Agente para Kaggriculture

Agente inteligente para la competición de Kaggle Kaggriculture.

## Estrategia

- **Fase temprana (días 1-5)**: Plantar trigo, comprar fertilizante.
- **Fase media (días 6-15)**: Diversificar con zanahorias y tomates, añadir gansos para huevos.
- **Fase tardía (días 16-30)**: Maximizar cultivos de alto valor (melón, fresa), comprar tierras, contratar trabajadores.

## Características

✅ Riego y fertilización automáticos  
✅ Cosecha en el momento óptimo  
✅ Compra/venta estratégica en el mercado  
✅ Gestión de animales (alimentación, cuidado, recolección)  
✅ Expansión de terreno  
✅ Contratación de trabajadores  

## Cómo usar

### Localmente

```bash
pip install kaggle-environments
python -c "
from kaggle_environments import make
env = make('kaggriculture', debug=True)
env.run(['main.py', 'random'])
print(env.steps[-1][0].reward)
"
