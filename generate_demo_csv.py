from app import make_demo_data
make_demo_data(3000).to_csv("synthetic_demo_data.csv", index=False)
print("Created synthetic_demo_data.csv (artificial data; not real customer data).")
