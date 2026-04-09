def calculate_depreciation(purchase_price, tire_cost, life_years,
                           method="straight_line", residual=42000, rate=0.144):
    if method == "straight_line":
        # Subtracting tires for precision in constant depreciation
        depreciable_base = purchase_price - (tire_cost + residual)
        annual_depreciation = depreciable_base / life_years
        written_down = [] #vehicle written down value after each year
        written_past = purchase_price - tire_cost #initial vehicle value
        for year in range(life_years):
            written_current = written_past - annual_depreciation
            written_down.append(written_current)
            written_past = written_current
        return ([annual_depreciation] * life_years, written_down)
    
    elif method == "declining_balance":
        # Usually calculated on the full initial cost
        schedule = []
        written_down = []
        written_past = purchase_price
        book_value = purchase_price
        for year in range(life_years):
            depreciation = book_value * rate
            schedule.append(round(depreciation, 0))
            book_value -= depreciation
            written_down.append(round(book_value, 0))
        return (schedule, written_down)

line = calculate_depreciation(125000, 7134, 7, "straight_line")
print(f"The constant depreciation is: {line}")

balance = calculate_depreciation(125000, 7134, 7, "declining_balance", 0.144)
print(f"The declining balance depreciation is: {balance}")
