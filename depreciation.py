def calculate_depreciation(purchase_price, tire_cost, life_years,
                           method="straight_line", written_down=42000, rate=0.144):
    if method == "straight_line":
        # Subtracting tires for precision in constant depreciation
        depreciable_base = purchase_price - tire_cost - written_down
        annual_depreciation = depreciable_base / life_years
        return [annual_depreciation] * life_years
    
    elif method == "declining_balance":
        # Usually calculated on the full initial cost
        schedule = []
        book_value = purchase_price
        for year in range(life_years):
            depreciation = book_value * rate
            schedule.append(round(depreciation, 0))
            book_value -= depreciation
        return schedule

line = calculate_depreciation(125000, 7134, 7, "straight_line")
print(f"The constant depreciation is: {line}")

balance = calculate_depreciation(125000, 7134, 7, "declining_balance", 0.144)
print(f"The declining balance depreciation is: {balance}")
