# Proyecto Titanic, Equipo 1. 
El proyecto corresponde al entregable 2 Modulo 3. Simular que el código pasa de experimentación (jupyter notebook) a un ambiente de desarrollo en python. 

# Funcionamiento
*Correr el módulo main.py* va a ejecutar los módulos secuencialmente:
- clean_split_data: lee datos desde la url y hace una pequeña limpieza de estos datos.  
- feature_engineering: prepara el train y el test set para que se pueda hacer un módelo de regresión logística sobre estos datos. 
- model_training: entrena el modelo y genera un archivo .pkl
- model_scoring: ejecuta el score del modelo. 

# Tecnologías
Algunas tecnologías utilizadas para el proyecto son:  
- **Para el empaquetado**
    - Setuptools
    - Virtual enviornment

- **Desarrollo y control de versiones**
    - Git / GitHub
    - Python

- **Para lectura y transformación de los datos** 
    - Pandas
    - Numpy
    - Scikit-learn (Pipelines personalizados)
    
- **Para buenas prácticas de documentación y cumplimiento del PEP8**
    - black
    - pycodestyle
    - pyment
    - Extension autoDocstring de VS Code

# Instalación del paquete
Se incluye un .wheel para instalar como un paquete. Aunque esto no tiene sentido en este contexto, se incluyó con fines didácticos.  
