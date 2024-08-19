class RandomGenerator(BaseGenerator):

    @AbstractGenerator._make_equation
    def _generate_random_expression(self, symbols: dict, allowed_operations:
                                    list, max_terms: int, **kwargs) -> sp.Eq:
        """Generates a random mathematical expression involving the provided symbols."""
        expression = 0
        num_terms = min(max_terms, len(symbols))
        selected_vars = random.sample(list(symbols.keys()), num_terms)

        used_operations = []

        for var in selected_vars:
            operation = self.rng.choice(allowed_operations)

            # Apply constraints on the usage of certain operations
            if operation == "log" and "log" not in used_operations:
                expression += sp.log(symbols[var] + 1)  # adjust log for computational stability
                used_operations.append("log")
            elif operation == "exp" and used_operations.count("exp") < 2:  # limit the number of exp used
                expression += sp.exp(symbols[var] % 3)  # Modulus to keep the exponent small
                used_operations.append("exp")
            elif operation in ["sin", "cos"] and used_operations.count(operation) < 2:
                expression += getattr(sp, operation)(symbols[var])
                used_operations.append(operation)
            elif operation in ["+", "-", "*", "/"]:
                coeff = self.rng.uniform(-5, 5)
                if operation == "/":
                    expression += symbols[var] / (coeff if coeff != 0 else 1)  # avoid division by zero
                else:
                    expression = sp.sympify(
                        f"{str(expression)}{operation}{coeff}*{var}",
                        evaluate=False
                    )

        return expression

