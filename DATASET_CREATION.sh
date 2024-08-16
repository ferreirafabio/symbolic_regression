python gpr/data/data_creator.py -f \
    dataloader.generator_type=PolynomialGenerator \
    dataloader.generator.num_nodes=2 \
    dataloader.generator.num_edges=2 \
    dataloader.generator.max_terms=4 \
    dataloader.generator.num_realizations=200 \
    dataloader.generator.allowed_operations="+,-,*,/,cos,log,sin,exp" \
    dataloader.generator.real_numbers_realizations=False \
    dataloader.generator.keep_graph=False \
    dataloader.generator.keep_data=False \
    dataloader.generator.real_numbers_variables=False \
    dataloader.generator.num_variables=2 \
    dataloader.generator.max_powers=1 \
    dataloader.generator.seed=1 \
    dataloader.train_samples=100000 \
    dataloader.valid_samples=50000 \
    dataloader.data_dir="/data/aadsqldb/symbolic_regression/polynomialgenerator"


python gpr/data/data_creator.py -f \
    dataloader.generator_type=PolynomialGenerator \
    dataloader.generator.num_nodes=2 \
    dataloader.generator.num_edges=2 \
    dataloader.generator.max_terms=4 \
    dataloader.generator.num_realizations=200 \
    dataloader.generator.allowed_operations="+,-,*,/,cos,log,sin,exp" \
    dataloader.generator.real_numbers_realizations=False \
    dataloader.generator.keep_graph=False \
    dataloader.generator.keep_data=False \
    dataloader.generator.real_numbers_variables=False \
    dataloader.generator.num_variables=2 \
    dataloader.generator.max_powers=1 \
    dataloader.generator.seed=1 \
    dataloader.train_samples=10000 \
    dataloader.valid_samples=5000 \
    dataloader.data_dir="/data/aadsqldb/symbolic_regression/polynomialgenerator"
