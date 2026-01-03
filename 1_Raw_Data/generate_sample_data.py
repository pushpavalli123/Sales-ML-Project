"""
Retail Sales Data Generator
Generates realistic Superstore-style retail sales data for analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
NUM_ORDERS = 1200
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 12, 31)

# Product catalog
PRODUCTS = {
    'Technology': {
        'Phones': [('iPhone 15 Pro', 999), ('Samsung Galaxy S24', 899), ('Google Pixel 8', 699), ('OnePlus 12', 799)],
        'Computers': [('MacBook Pro 14"', 1999), ('Dell XPS 15', 1499), ('HP Spectre x360', 1299), ('Lenovo ThinkPad', 1199)],
        'Accessories': [('Apple AirPods Pro', 249), ('Logitech MX Master', 99), ('Samsung T7 SSD', 129), ('Anker PowerBank', 49)]
    },
    'Furniture': {
        'Chairs': [('Herman Miller Aeron', 1395), ('Secretlab Titan', 449), ('IKEA Markus', 229), ('Steelcase Leap', 1199)],
        'Tables': [('Standing Desk Pro', 599), ('IKEA BEKANT', 349), ('Uplift V2', 699), ('Autonomous SmartDesk', 499)],
        'Bookcases': [('IKEA Billy', 79), ('Pottery Barn Ladder', 299), ('West Elm Industrial', 449), ('Target Threshold', 129)]
    },
    'Office Supplies': {
        'Paper': [('HP Printer Paper', 24.99), ('Hammermill Copy Plus', 34.99), ('Staples Multipurpose', 29.99), ('Amazon Basics Paper', 19.99)],
        'Binders': [('Avery Durable Binder', 8.99), ('Cardinal Economy', 5.99), ('Staples Better Binder', 12.99), ('Wilson Jones', 7.99)],
        'Art Supplies': [('Prismacolor Pencils', 29.99), ('Crayola Markers', 14.99), ('Sharpie Fine Point', 19.99), ('Paper Mate Pens', 9.99)],
        'Storage': [('Bankers Box', 24.99), ('Sterilite Organizer', 15.99), ('IRIS File Box', 19.99), ('Rubbermaid Container', 12.99)]
    }
}

# Customer segments
SEGMENTS = ['Consumer', 'Corporate', 'Home Office']

# Regions and states
REGIONS = {
    'East': ['New York', 'Pennsylvania', 'Massachusetts', 'New Jersey', 'Connecticut'],
    'West': ['California', 'Washington', 'Oregon', 'Colorado', 'Arizona'],
    'Central': ['Texas', 'Illinois', 'Ohio', 'Michigan', 'Minnesota'],
    'South': ['Florida', 'Georgia', 'North Carolina', 'Virginia', 'Tennessee']
}

# Cities per state
CITIES = {
    'New York': ['New York City', 'Buffalo', 'Rochester', 'Albany'],
    'California': ['Los Angeles', 'San Francisco', 'San Diego', 'Sacramento'],
    'Texas': ['Houston', 'Dallas', 'Austin', 'San Antonio'],
    'Florida': ['Miami', 'Orlando', 'Tampa', 'Jacksonville'],
    'Illinois': ['Chicago', 'Springfield', 'Naperville'],
    'Pennsylvania': ['Philadelphia', 'Pittsburgh', 'Harrisburg'],
    'Washington': ['Seattle', 'Spokane', 'Tacoma'],
    'Georgia': ['Atlanta', 'Savannah', 'Augusta'],
    'Massachusetts': ['Boston', 'Cambridge', 'Worcester'],
    'Ohio': ['Columbus', 'Cleveland', 'Cincinnati'],
    'North Carolina': ['Charlotte', 'Raleigh', 'Durham'],
    'Michigan': ['Detroit', 'Grand Rapids', 'Ann Arbor'],
    'New Jersey': ['Newark', 'Jersey City', 'Trenton'],
    'Virginia': ['Richmond', 'Virginia Beach', 'Norfolk'],
    'Oregon': ['Portland', 'Salem', 'Eugene'],
    'Colorado': ['Denver', 'Boulder', 'Colorado Springs'],
    'Arizona': ['Phoenix', 'Tucson', 'Scottsdale'],
    'Minnesota': ['Minneapolis', 'St. Paul', 'Rochester'],
    'Tennessee': ['Nashville', 'Memphis', 'Knoxville'],
    'Connecticut': ['Hartford', 'New Haven', 'Stamford']
}

# Ship modes
SHIP_MODES = ['Standard Class', 'Second Class', 'First Class', 'Same Day']
SHIP_DAYS = {'Standard Class': (5, 7), 'Second Class': (3, 5), 'First Class': (2, 3), 'Same Day': (0, 0)}

# Generate customer names
FIRST_NAMES = ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Emily', 'Robert', 'Lisa', 'William', 'Jennifer',
               'James', 'Amanda', 'Christopher', 'Jessica', 'Daniel', 'Ashley', 'Matthew', 'Brittany', 'Anthony', 'Stephanie',
               'Mark', 'Nicole', 'Brian', 'Heather', 'Steven', 'Michelle', 'Paul', 'Kimberly', 'Andrew', 'Melissa',
               'Rajesh', 'Priya', 'Amit', 'Sneha', 'Vikram', 'Anita', 'Sanjay', 'Pooja', 'Arun', 'Divya']

LAST_NAMES = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez',
              'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin',
              'Patel', 'Sharma', 'Singh', 'Kumar', 'Gupta', 'Shah', 'Reddy', 'Iyer', 'Nair', 'Rao']

def generate_customer_id():
    return f"CUS-{random.randint(10000, 99999)}"

def generate_order_id(index):
    return f"ORD-{2023 + index // 600}-{str(index % 10000).zfill(5)}"

def generate_product_id(category, subcategory, index):
    cat_prefix = category[:3].upper()
    sub_prefix = subcategory[:3].upper()
    return f"{cat_prefix}-{sub_prefix}-{str(index).zfill(4)}"

def random_date(start, end):
    delta = end - start
    random_days = random.randint(0, delta.days)
    return start + timedelta(days=random_days)

def calculate_shipping_cost(category, ship_mode, quantity):
    base_costs = {'Technology': 15, 'Furniture': 50, 'Office Supplies': 5}
    mode_multiplier = {'Standard Class': 1, 'Second Class': 1.5, 'First Class': 2, 'Same Day': 3}
    return round(base_costs[category] * mode_multiplier[ship_mode] * (1 + quantity * 0.1), 2)

def calculate_profit_margin(category, discount):
    # Base margins by category
    base_margins = {'Technology': 0.15, 'Furniture': 0.25, 'Office Supplies': 0.35}
    # Discount reduces margin
    margin = base_margins[category] - (discount * 0.5)
    return max(margin, -0.1)  # Can have negative margin with high discounts

# Generate customers (100 unique customers)
customers = []
for i in range(100):
    customers.append({
        'customer_id': generate_customer_id(),
        'customer_name': f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
        'segment': random.choice(SEGMENTS)
    })

# Generate orders
orders = []
for i in range(NUM_ORDERS):
    # Pick random category, subcategory, product
    category = random.choice(list(PRODUCTS.keys()))
    subcategory = random.choice(list(PRODUCTS[category].keys()))
    product_name, unit_price = random.choice(PRODUCTS[category][subcategory])
    
    # Pick random region, state, city
    region = random.choice(list(REGIONS.keys()))
    state = random.choice(REGIONS[region])
    city = random.choice(CITIES[state])
    
    # Pick random customer
    customer = random.choice(customers)
    
    # Generate dates
    order_date = random_date(START_DATE, END_DATE)
    ship_mode = random.choices(SHIP_MODES, weights=[0.6, 0.2, 0.15, 0.05])[0]
    ship_days = random.randint(*SHIP_DAYS[ship_mode])
    ship_date = order_date + timedelta(days=ship_days)
    
    # Generate quantities and discounts
    quantity = random.choices([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
                              weights=[0.3, 0.25, 0.15, 0.1, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01])[0]
    
    # Discount based on quantity and segment
    if customer['segment'] == 'Corporate':
        discount = random.choices([0, 0.05, 0.1, 0.15, 0.2, 0.25], weights=[0.2, 0.25, 0.25, 0.15, 0.1, 0.05])[0]
    elif customer['segment'] == 'Home Office':
        discount = random.choices([0, 0.05, 0.1, 0.15, 0.2], weights=[0.3, 0.3, 0.2, 0.15, 0.05])[0]
    else:
        discount = random.choices([0, 0.05, 0.1, 0.15], weights=[0.4, 0.3, 0.2, 0.1])[0]
    
    # Calculate sales, shipping, profit
    sales = round(unit_price * quantity * (1 - discount), 2)
    shipping_cost = calculate_shipping_cost(category, ship_mode, quantity)
    profit_margin = calculate_profit_margin(category, discount)
    profit = round(sales * profit_margin - shipping_cost * 0.1, 2)
    
    order = {
        'order_id': generate_order_id(i),
        'order_date': order_date.strftime('%Y-%m-%d'),
        'ship_date': ship_date.strftime('%Y-%m-%d'),
        'ship_mode': ship_mode,
        'customer_id': customer['customer_id'],
        'customer_name': customer['customer_name'],
        'segment': customer['segment'],
        'country': 'United States',
        'city': city,
        'state': state,
        'region': region,
        'product_id': generate_product_id(category, subcategory, i % 100),
        'category': category,
        'sub_category': subcategory,
        'product_name': product_name,
        'quantity': quantity,
        'unit_price': unit_price,
        'discount': discount,
        'sales': sales,
        'profit': profit,
        'shipping_cost': shipping_cost
    }
    orders.append(order)

# Create DataFrame
df = pd.DataFrame(orders)

# Sort by order date
df = df.sort_values('order_date').reset_index(drop=True)

# Save to CSV
output_path = os.path.join(os.path.dirname(__file__), 'retail_sales_data.csv')
df.to_csv(output_path, index=False)

print(f"✅ Generated {len(df)} orders successfully!")
print(f"📁 Saved to: {output_path}")
print("\n📊 Dataset Summary:")
print(f"   • Date Range: {df['order_date'].min()} to {df['order_date'].max()}")
print(f"   • Total Sales: ${df['sales'].sum():,.2f}")
print(f"   • Total Profit: ${df['profit'].sum():,.2f}")
print(f"   • Unique Customers: {df['customer_id'].nunique()}")
print(f"   • Categories: {df['category'].nunique()}")
print(f"   • Regions: {df['region'].nunique()}")
print("\n📋 Columns:")
print(df.columns.tolist())
print("\n🔍 Sample Data:")
print(df.head())
