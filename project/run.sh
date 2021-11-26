#!/bin/bash
if test $# -ne 1; then
    echo "Must give me the binary for running the tests"
else
    echo "Running flower grayscale:\n"
    $1 ../images/flower.png --grayscale < ./filters/blur_filter.txt
    $1 ../images/flower.png --grayscale < ./filters/blur_filter5.txt

    echo "Running flower:\n"
    $1 ../images/flower.png < ./filters/blur_filter.txt
    $1 ../images/flower.png < ./filters/blur_filter5.txt

    echo "Running Vangogh grayscale:\n"
    $1 ../images/Vancock.jpeg --grayscale < ./filters/blur_filter.txt
    $1 ../images/Vancock.jpeg --grayscale < ./filters/blur_filter5.txt

    echo "Running Vangogh:\n"
    $1 ../images/Vancock.jpeg < ./filters/blur_filter.txt
    $1 ../images/Vancock.jpeg < ./filters/blur_filter5.txt

    echo "Running Appa grayscale:\n"
    $1 ../images/appa.jpg --grayscale < ./filters/blur_filter.txt
    $1 ../images/appa.jpg --grayscale < ./filters/blur_filter5.txt

    echo "Running Appa:\n"
    $1 ../images/appa.jpg < ./filters/blur_filter.txt
    $1 ../images/appa.jpg < ./filters/blur_filter5.txt

    echo "Running Dino grayscale:\n"
    $1 ../images/dino.png --grayscale < ./filters/blur_filter.txt
    $1 ../images/dino.png --grayscale < ./filters/blur_filter5.txt

    echo "Running Appa:\n"
    $1 ../images/dino.png < ./filters/blur_filter.txt
    $1 ../images/dino.png < ./filters/blur_filter5.txt
fi
