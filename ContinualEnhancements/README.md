# Continual Enhancements

Welcome to the **Continual Enhancements** module!  
This directory contains methods designed to complement any unlearning technique, with a focus on **preventing model degradation** (i.e., unintentional unlearning).

## Overview

- **Plug-and-Play:**  
    Easily add new enhancement methods here. Each method can be seamlessly integrated and invoked from the `UnlearningMethods` module.

- **Merge Functionality:**  
    The only method directly called from this module is `Merge`, which efficiently combines unlearning checkpoints to maintain model integrity.

## How to Use

1. **Add Your Method:**  
     Implement your enhancement method in this directory.

2. **Integrate:**  
     Call your method from `UnlearningMethods` as needed.
---

Feel free to contribute new methods or improvements!