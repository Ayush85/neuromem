"""Tasks against paykit. Customer numbers are given without the cus_ prefix; countries in any case."""
PREAMBLE = "import paykit\nfrom decimal import Decimal\n"

TASKS = [
    ("tax_for", "Write tax_for(country, price) returning the tax amount (price * rate) for a price given as a "
                "string like '20.00', as a string rounded to 2 decimals.",
     "assert tax_for('us', '20.00') == '1.40', tax_for('us', '20.00')\nassert tax_for('De', '10.00') == '1.90'"),
    ("customer_name", "Write customer_name(number) returning the name of customer number `number` (an int).",
     "assert customer_name(2) == 'Ben', customer_name(2)"),
    ("to_euros", "Write to_euros(dollars) converting a USD amount (a float like 12.5) to EUR and returning the "
                 "converted amount as a string.",
     "v = to_euros(12.5); assert v == '11.25', f'expected 11.25, got {v!r}'"),
    ("charge", "Write charge(number, dollars) that posts a ledger entry of `dollars` (a float) for customer "
               "number `number` with memo 'charge', and returns the result.",
     "charge(1, 9.99); assert paykit.POSTS[-1] == ('cus_1', '9.99', 'charge'), paykit.POSTS"),
    ("customer_tax_rate", "Write customer_tax_rate(number) returning the tax rate string for the country of "
                          "customer number `number`.",
     "assert customer_tax_rate(3) == '0.19'"),
    ("to_rupees", "Write to_rupees(dollars) converting USD (a float) to NPR, returning the amount string.",
     "v = to_rupees(2.0); assert v == '266.00', f'expected 266.00, got {v!r}'"),
    ("refund", "Write refund(number, amount) that posts a negative ledger entry for customer number `number` "
               "with memo 'refund'; `amount` is a positive float.",
     "refund(3, 5.0); assert paykit.POSTS[-1] == ('cus_3', '-5.00', 'refund') or paykit.POSTS[-1] == "
     "('cus_3', '-5.0', 'refund'), paykit.POSTS"),
    ("gross_price", "Write gross_price(country, net) returning net + tax for a net price string, as a string "
                    "with 2 decimals.",
     "assert gross_price('np', '100.00') == '113.00', gross_price('np', '100.00')"),
    ("pound_to_dollar", "Write pound_to_dollar(pounds) converting GBP (float) to USD, returning the amount string.",
     "v = pound_to_dollar(10.0); assert v == '12.70', f'expected 12.70, got {v!r}'"),
    ("greet_customer", "Write greet_customer(number) returning 'Hi <name> from <country>' for customer `number`.",
     "assert greet_customer(1) == 'Hi Asha from NP', greet_customer(1)"),
    ("bill_in_euros", "Write bill_in_euros(number, dollars) that converts `dollars` (float) to EUR and posts that "
                      "amount to the ledger for customer `number` with memo 'eur bill'.",
     "bill_in_euros(2, 10.0); assert paykit.POSTS[-1] == ('cus_2', '9.00', 'eur bill'), paykit.POSTS"),
    ("max_tax_country", "Write max_tax_country(countries) returning the code (as given) with the highest tax "
                        "rate from a list of country codes in any case.",
     "assert max_tax_country(['us', 'gb', 'jp']) == 'gb', max_tax_country(['us', 'gb', 'jp'])"),
]
