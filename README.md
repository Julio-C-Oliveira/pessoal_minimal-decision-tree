Esse repositório vai servir para estudar e desenvolver um modelo de árvore leve, que rode em dispositivos bem limitados, a ideia inicial e conseuir rodar em dispositivos que possuem 1MB de RAM.

**TO-DO:**
- [ ] Estudar sobre Hoeffding Tree:
    -[ ] Mining High-Speed Data Streams - Pedro Domingos e Geoff Hulten. Apresenta a ideia de desigualdade de hoeffding para fazer os splits.
    - [ ] Very Fast Decision Tree Learner - Hulten, Spencer e Domingos. Melhorias em consumo de memória.
    - [] Adaptive Strategies for Learning Classifiers from Data Streams - Albert Bifet. Fala sobre o tratamento de dados categóricos e numéricos.
    - [ ] Learning from Time-Changing Data Streams with Adaptive Hoeffding Trees - Albert Bifet e Ricard Gavaldà. Fala sobre COncept Drift.
- [ ] Estudar a implementação do River (https://github.com/online-ml/river):
    - [ ] /river/tree
- [ ] Iniciar a implementação em C.

# Hoeffding Tree
Vou utilizar esse algoritmo no lugar do CART, pois o CART precisa de todos os dados carregados em RAM, já o Hoeffding não.
