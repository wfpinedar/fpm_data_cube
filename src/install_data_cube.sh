#!/bin/bash
if [ "$DEBUG_MODE" == "true" ]
then
  echo 'Modo Depuración Habilitado'
  
  curl https://raw.githubusercontent.com/wfpinedar/fpm_data_cube/master/utils -o /tmp/utils
  . /tmp/utils
  
  curl https://raw.githubusercontent.com/wfpinedar/fpm_data_cube/master/src/bitacora.sh -o /tmp/bitacora.sh
  while read -r linea
  do
    printc "$linea"
    #eval "$linea"
    $linea
  done < /tmp/bitacora.sh
 
else
  echo 'Modo Depuración Deshabilitado'
  curl https://raw.githubusercontent.com/wfpinedar/fpm_data_cube/master/src/bitacora.sh | sh
fi
