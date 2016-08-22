#Correr script manualmente

Para correr el script basta con ejecutar en una terminal el comando:

```bash
$ curl https://raw.githubusercontent.com/wfpinedar/fpm_data_cube/master/src/install_data_cube.sh | sudo sh
```

O como root:

```bash
# curl https://raw.githubusercontent.com/wfpinedar/fpm_data_cube/master/src/install_data_cube.sh | sh
```

Para modo depuración:
```bash
$ curl https://raw.githubusercontent.com/wfpinedar/fpm_data_cube/master/src/install_data_cube.sh -o install_data_cube.sh
$ chmod +x install_data_cube.sh
$ DEBUG_MODE=true ./install_data_cube.sh
```
