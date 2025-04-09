# ML_FPGA_UNMdP


## Windows
- Descargar Python 3.9 de la *Microsoft Store*
- Una vez instalado, crear un entorno virtual desde CMD

```{cmd}
> python3.9 -m venv <path\donde\se\quiera\instalar>
```

- Iniciar el entorno virtual ejecutando el comando:
```{cmd}
> <path\donde\se\haya\instalado>\bin\activate.bat
```

- Debería verse de la siguiente forma:

```{cmd}
(.nombreDelVenv) > 
```

- Actualizar pip:
```{cmd}
(.nombreDelVenv) > python -m pip install --upgrade pip
```

- Instalar los paquetes requeridos (descargar/utilizar archivo `requirements__py3.9.13.txt`):
```{cmd}
(.nombreDelVenv) > python -m pip install -r <path/al/archivo/requirements/descargado>
```
