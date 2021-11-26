#!/bin/bash
#*----------------------------------------------------------------
# Programación avanzada: Proyecto final
# Fecha: 25-Nov-2021
# Autor: A01651517 Pedro González
#--------------------------------------------------------------*/
if test $# -ne 1; then
    echo "Must give me the binary for running the tests"
else
#    echo "Running flower gray:"
#    $1 ../images/flower.png --gray < ./filters/blur_filter.txt
#    $1 ../images/flower.png --gray < ./filters/blur_filter5.txt

    echo "Running flower:"
    $1 ../images/flower.png < ./filters/blur_filter.txt
    $1 ../images/flower.png < ./filters/blur_filter5.txt

#    echo "Running Vangogh gray:"
#    $1 ../images/Vancock.jpeg --gray < ./filters/blur_filter.txt
#    $1 ../images/Vancock.jpeg --gray < ./filters/blur_filter5.txt

    echo "Running Vangogh:"
    $1 ../images/Vancock.jpeg < ./filters/blur_filter.txt
    $1 ../images/Vancock.jpeg < ./filters/blur_filter5.txt

#    echo "Running Appa gray:"
#    $1 ../images/appa.jpg --gray < ./filters/blur_filter.txt
#    $1 ../images/appa.jpg --gray < ./filters/blur_filter5.txt

    echo "Running Appa:"
    $1 ../images/appa.jpg < ./filters/blur_filter.txt
    $1 ../images/appa.jpg < ./filters/blur_filter5.txt

#    echo "Running Dino gray:"
#    $1 ../images/dino.png --gray < ./filters/blur_filter.txt
#    $1 ../images/dino.png --gray < ./filters/blur_filter5.txt

    echo "Running Appa:"
    $1 ../images/dino.png < ./filters/blur_filter.txt
    $1 ../images/dino.png < ./filters/blur_filter5.txt
fi
