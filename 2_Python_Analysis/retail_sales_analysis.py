"""
Retail Sales Performance Analysis
Complete EDA, Data Cleaning, and Visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Color palette
COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#44AF69']

print("=" * 60)
print("    RETAIL SALES PERFORMANCE ANALYSIS")
print("=" * 60)

# ============================================================
# STEP 1: DATA LOADING
# ============================================================
print("\n📂 STEP 1: Loading Data...")

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_path = os.path.join(project_root, '1_Raw_Data', 'retail_sales_data.csv')

df = pd.read_csv(data_path)

print(f"   ✓ Loaded {len(df)} records from retail_sales_data.csv")
print(f"\n📋 Dataset Shape: {df.shape}")
print(f"\n📋 Columns ({len(df.columns)}):")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2}. {col}")

# ============================================================
# STEP 2: DATA INSPECTION
# ============================================================
print("\n" + "=" * 60)
print("📊 STEP 2: Data Inspection")
print("=" * 60)

print("\n🔍 Data Types:")
print(df.dtypes)

print("\n📈 Statistical Summary (Numeric Columns):")
print(df.describe().round(2))

print("\n📈 Statistical Summary (Categorical Columns):")
print(df.describe(include=['object']))

# ============================================================
# STEP 3: DATA CLEANING
# ============================================================
print("\n" + "=" * 60)
print("🧹 STEP 3: Data Cleaning")
print("=" * 60)

# Check for missing values
print("\n🔍 Missing Values:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "   ✓ No missing values found!")

# Check for duplicates
duplicates = df.duplicated(subset=['order_id']).sum()
print(f"\n🔍 Duplicate Order IDs: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates(subset=['order_id'])
    print(f"   ✓ Removed {duplicates} duplicate orders")

# Convert date columns
df['order_date'] = pd.to_datetime(df['order_date'])
df['ship_date'] = pd.to_datetime(df['ship_date'])
print("   ✓ Converted date columns to datetime")

# Create derived columns
df['year'] = df['order_date'].dt.year
df['month'] = df['order_date'].dt.month
df['month_name'] = df['order_date'].dt.strftime('%b')
df['year_month'] = df['order_date'].dt.strftime('%Y-%m')
df['quarter'] = df['order_date'].dt.quarter
df['day_of_week'] = df['order_date'].dt.day_name()
df['shipping_days'] = (df['ship_date'] - df['order_date']).dt.days
df['profit_margin'] = (df['profit'] / df['sales'] * 100).round(2)
df['discount_amount'] = (df['sales'] * df['discount'] / (1 - df['discount'])).round(2)

print("   ✓ Created derived columns: year, month, quarter, day_of_week, shipping_days, profit_margin, discount_amount")

print(f"\n✅ Cleaned Data Shape: {df.shape}")

# ============================================================
# STEP 4: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n" + "=" * 60)
print("📊 STEP 4: Exploratory Data Analysis")
print("=" * 60)

# Key Metrics
total_sales = df['sales'].sum()
total_profit = df['profit'].sum()
avg_order_value = df['sales'].mean()
order_count = len(df)
unique_customers = df['customer_id'].nunique()
profit_margin_pct = (total_profit / total_sales * 100)

print("\n" + "─" * 40)
print("             KEY METRICS")
print("─" * 40)
print(f"   💰 Total Revenue:       ${total_sales:>15,.2f}")
print(f"   📈 Total Profit:        ${total_profit:>15,.2f}")
print(f"   📊 Profit Margin:       {profit_margin_pct:>15.2f}%")
print(f"   🛒 Total Orders:        {order_count:>15,}")
print(f"   👥 Unique Customers:    {unique_customers:>15,}")
print(f"   💵 Avg Order Value:     ${avg_order_value:>15,.2f}")
print("─" * 40)

# Sales by Region
print("\n📍 SALES BY REGION:")
sales_by_region = df.groupby('region').agg({
    'sales': 'sum',
    'profit': 'sum',
    'order_id': 'count'
}).rename(columns={'order_id': 'orders'}).sort_values('sales', ascending=False)
sales_by_region['profit_margin'] = (sales_by_region['profit'] / sales_by_region['sales'] * 100).round(2)
print(sales_by_region.round(2))

# Sales by Category
print("\n📦 SALES BY CATEGORY:")
sales_by_category = df.groupby('category').agg({
    'sales': 'sum',
    'profit': 'sum',
    'order_id': 'count'
}).rename(columns={'order_id': 'orders'}).sort_values('sales', ascending=False)
sales_by_category['profit_margin'] = (sales_by_category['profit'] / sales_by_category['sales'] * 100).round(2)
print(sales_by_category.round(2))

# Sales by Segment
print("\n👥 SALES BY CUSTOMER SEGMENT:")
sales_by_segment = df.groupby('segment').agg({
    'sales': 'sum',
    'profit': 'sum',
    'order_id': 'count'
}).rename(columns={'order_id': 'orders'}).sort_values('sales', ascending=False)
sales_by_segment['profit_margin'] = (sales_by_segment['profit'] / sales_by_segment['sales'] * 100).round(2)
print(sales_by_segment.round(2))

# Monthly Sales Trend
print("\n📅 MONTHLY SALES TREND (2024):")
monthly_sales_2024 = df[df['year'] == 2024].groupby('month_name').agg({
    'sales': 'sum',
    'profit': 'sum'
}).round(2)
# Reindex to proper month order
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_sales_2024 = monthly_sales_2024.reindex([m for m in month_order if m in monthly_sales_2024.index])
print(monthly_sales_2024)

# Top 10 Products
print("\n🏆 TOP 10 PRODUCTS BY SALES:")
top_products = df.groupby('product_name')['sales'].sum().sort_values(ascending=False).head(10)
for i, (product, sales) in enumerate(top_products.items(), 1):
    print(f"   {i:2}. {product:<25} ${sales:>12,.2f}")

# Top 10 Customers
print("\n🌟 TOP 10 CUSTOMERS BY TOTAL SPENDING:")
top_customers = df.groupby('customer_name').agg({
    'sales': 'sum',
    'order_id': 'count'
}).rename(columns={'order_id': 'orders'}).sort_values('sales', ascending=False).head(10)
print(top_customers.round(2))

# Day of Week Analysis
print("\n📆 SALES BY DAY OF WEEK:")
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
sales_by_day = df.groupby('day_of_week')['sales'].sum().reindex(day_order)
for day, sales in sales_by_day.items():
    print(f"   {day:<12} ${sales:>12,.2f}")

# Ship Mode Analysis
print("\n🚚 SALES BY SHIP MODE:")
ship_mode_analysis = df.groupby('ship_mode').agg({
    'sales': 'sum',
    'shipping_days': 'mean',
    'order_id': 'count'
}).rename(columns={'order_id': 'orders', 'shipping_days': 'avg_ship_days'}).sort_values('sales', ascending=False)
print(ship_mode_analysis.round(2))

# Discount Impact Analysis
print("\n💸 DISCOUNT IMPACT ANALYSIS:")
df['discount_group'] = pd.cut(df['discount'], bins=[-0.01, 0, 0.1, 0.2, 1], 
                               labels=['No Discount', '1-10%', '11-20%', '20%+'])
discount_analysis = df.groupby('discount_group').agg({
    'sales': 'sum',
    'profit': 'sum',
    'order_id': 'count'
}).rename(columns={'order_id': 'orders'})
discount_analysis['avg_profit_margin'] = (discount_analysis['profit'] / discount_analysis['sales'] * 100).round(2)
print(discount_analysis.round(2))

# ============================================================
# STEP 5: VISUALIZATIONS
# ============================================================
print("\n" + "=" * 60)
print("📊 STEP 5: Creating Visualizations")
print("=" * 60)

# Create output directory for visualizations
viz_dir = script_dir
os.makedirs(viz_dir, exist_ok=True)

# 1. Sales by Region (Bar Chart)
fig, ax = plt.subplots(figsize=(10, 6))
region_sales = df.groupby('region')['sales'].sum().sort_values(ascending=True)
bars = ax.barh(region_sales.index, region_sales.values, color=COLORS[:4])
ax.set_xlabel('Sales Amount ($)')
ax.set_title('Total Sales by Region', fontweight='bold', fontsize=16)
for bar, value in zip(bars, region_sales.values):
    ax.text(value + 5000, bar.get_y() + bar.get_height()/2, f'${value:,.0f}', 
            va='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'sales_by_region.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: sales_by_region.png")

# 2. Sales by Category (Pie Chart)
fig, ax = plt.subplots(figsize=(10, 8))
category_sales = df.groupby('category')['sales'].sum()
wedges, texts, autotexts = ax.pie(category_sales.values, labels=category_sales.index, 
                                   autopct='%1.1f%%', colors=COLORS[:3],
                                   explode=[0.02, 0.02, 0.02], shadow=True,
                                   textprops={'fontsize': 12})
for autotext in autotexts:
    autotext.set_fontweight('bold')
ax.set_title('Sales Distribution by Category', fontweight='bold', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'sales_by_category.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: sales_by_category.png")

# 3. Monthly Sales Trend (Line Chart)
fig, ax = plt.subplots(figsize=(14, 6))
monthly_trend = df.groupby('year_month')['sales'].sum().reset_index()
ax.plot(monthly_trend['year_month'], monthly_trend['sales'], 
        marker='o', linewidth=2, markersize=6, color=COLORS[0])
ax.fill_between(monthly_trend['year_month'], monthly_trend['sales'], alpha=0.3, color=COLORS[0])
ax.set_xlabel('Month')
ax.set_ylabel('Sales Amount ($)')
ax.set_title('Monthly Sales Trend (2023-2024)', fontweight='bold', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'monthly_sales_trend.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: monthly_sales_trend.png")

# 4. Sales vs Profit by Category (Grouped Bar Chart)
fig, ax = plt.subplots(figsize=(10, 6))
cat_data = df.groupby('category').agg({'sales': 'sum', 'profit': 'sum'}).reset_index()
x = np.arange(len(cat_data))
width = 0.35
bars1 = ax.bar(x - width/2, cat_data['sales'], width, label='Sales', color=COLORS[0])
bars2 = ax.bar(x + width/2, cat_data['profit'], width, label='Profit', color=COLORS[3])
ax.set_ylabel('Amount ($)')
ax.set_title('Sales vs Profit by Category', fontweight='bold', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(cat_data['category'])
ax.legend()
ax.bar_label(bars1, fmt='${:,.0f}', padding=3, fontsize=8)
ax.bar_label(bars2, fmt='${:,.0f}', padding=3, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'sales_vs_profit_by_category.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: sales_vs_profit_by_category.png")

# 5. Top 10 Products (Horizontal Bar Chart)
fig, ax = plt.subplots(figsize=(12, 8))
top_products_df = df.groupby('product_name')['sales'].sum().sort_values(ascending=True).tail(10)
bars = ax.barh(top_products_df.index, top_products_df.values, color=COLORS[1])
ax.set_xlabel('Sales Amount ($)')
ax.set_title('Top 10 Products by Sales', fontweight='bold', fontsize=16)
for bar, value in zip(bars, top_products_df.values):
    ax.text(value + 2000, bar.get_y() + bar.get_height()/2, f'${value:,.0f}', 
            va='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'top_10_products.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: top_10_products.png")

# 6. Customer Segment Analysis (Donut Chart)
fig, ax = plt.subplots(figsize=(10, 8))
segment_sales = df.groupby('segment')['sales'].sum()
wedges, texts, autotexts = ax.pie(segment_sales.values, labels=segment_sales.index, 
                                   autopct='%1.1f%%', colors=COLORS[:3],
                                   pctdistance=0.75, textprops={'fontsize': 12})
centre_circle = plt.Circle((0, 0), 0.50, fc='white')
ax.add_patch(centre_circle)
ax.set_title('Sales by Customer Segment', fontweight='bold', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'sales_by_segment.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: sales_by_segment.png")

# 7. Quarterly Sales Comparison (Bar Chart)
fig, ax = plt.subplots(figsize=(12, 6))
quarterly_sales = df.groupby(['year', 'quarter'])['sales'].sum().unstack(level=0)
quarterly_sales.plot(kind='bar', ax=ax, color=[COLORS[0], COLORS[2]], width=0.7)
ax.set_xlabel('Quarter')
ax.set_ylabel('Sales Amount ($)')
ax.set_title('Quarterly Sales Comparison (2023 vs 2024)', fontweight='bold', fontsize=16)
ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'], rotation=0)
ax.legend(title='Year')
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'quarterly_sales_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: quarterly_sales_comparison.png")

# 8. Profit Margin by Region Heatmap
fig, ax = plt.subplots(figsize=(12, 8))
pivot_data = df.pivot_table(values='profit_margin', index='category', columns='region', aggfunc='mean')
sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=ax,
            linewidths=0.5, cbar_kws={'label': 'Profit Margin (%)'})
ax.set_title('Average Profit Margin by Category and Region', fontweight='bold', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'profit_margin_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: profit_margin_heatmap.png")

# 9. Day of Week Sales (Bar Chart)
fig, ax = plt.subplots(figsize=(10, 6))
day_sales = df.groupby('day_of_week')['sales'].sum().reindex(day_order)
bars = ax.bar(day_sales.index, day_sales.values, color=COLORS[4])
ax.set_xlabel('Day of Week')
ax.set_ylabel('Sales Amount ($)')
ax.set_title('Sales by Day of Week', fontweight='bold', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'sales_by_day_of_week.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: sales_by_day_of_week.png")

# 10. Discount Impact on Profit (Scatter Plot)
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(df['discount']*100, df['profit_margin'], 
                     c=df['sales'], cmap='viridis', alpha=0.6, s=50)
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, label='Break-even')
ax.set_xlabel('Discount (%)')
ax.set_ylabel('Profit Margin (%)')
ax.set_title('Discount Impact on Profit Margin', fontweight='bold', fontsize=16)
plt.colorbar(scatter, label='Sales Amount ($)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'discount_impact_scatter.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: discount_impact_scatter.png")

# ============================================================
# STEP 6: KEY INSIGHTS SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("💡 STEP 6: KEY INSIGHTS & FINDINGS")
print("=" * 60)

# Best & Worst Performers
best_region = df.groupby('region')['sales'].sum().idxmax()
best_category = df.groupby('category')['sales'].sum().idxmax()
best_product = df.groupby('product_name')['sales'].sum().idxmax()
best_customer = df.groupby('customer_name')['sales'].sum().idxmax()

worst_region_profit = df.groupby('region')['profit'].sum().idxmin()
worst_category_margin = (df.groupby('category')['profit'].sum() / df.groupby('category')['sales'].sum()).idxmin()

print(f"""
┌──────────────────────────────────────────────────────────┐
│                    KEY INSIGHTS                          │
├──────────────────────────────────────────────────────────┤
│  📍 REGIONAL INSIGHTS:                                   │
│     • Best Region: {best_region:<20} (Highest Sales)    │
│     • Region needing attention: {worst_region_profit:<15} (Lowest Profit) │
│                                                          │
│  📦 CATEGORY INSIGHTS:                                   │
│     • Top Category: {best_category:<20}                  │
│     • Lowest Margin: {worst_category_margin:<20}         │
│                                                          │
│  🏆 TOP PERFORMERS:                                      │
│     • Best Product: {best_product[:25]:<25}              │
│     • Best Customer: {best_customer[:20]:<20}            │
│                                                          │
│  💡 RECOMMENDATIONS:                                     │
│     1. Focus marketing on {best_region} region           │
│     2. Review pricing strategy for {worst_category_margin}│
│     3. Implement loyalty program for top customers       │
│     4. Reduce high discounts (>20%) to improve margins   │
│     5. Optimize shipping costs in low-profit regions     │
└──────────────────────────────────────────────────────────┘
""")

# ============================================================
# STEP 7: EXPORT CLEANED DATA
# ============================================================
print("\n" + "=" * 60)
print("📤 STEP 7: Exporting Data")
print("=" * 60)

# Export cleaned data for Power BI
power_bi_dir = os.path.join(project_root, '3_Power_BI')
os.makedirs(power_bi_dir, exist_ok=True)

df.to_csv(os.path.join(power_bi_dir, 'cleaned_sales_data.csv'), index=False)
print(f"   ✓ Exported cleaned data to: 3_Power_BI/cleaned_sales_data.csv")

# Also save as Excel for Power BI
df.to_excel(os.path.join(power_bi_dir, 'cleaned_sales_data.xlsx'), index=False, engine='openpyxl')
print(f"   ✓ Exported Excel file to: 3_Power_BI/cleaned_sales_data.xlsx")

# Export summary statistics
summary_stats = {
    'Metric': ['Total Sales', 'Total Profit', 'Profit Margin %', 'Total Orders', 
               'Unique Customers', 'Average Order Value', 'Total Products', 
               'Analysis Date'],
    'Value': [f"${total_sales:,.2f}", f"${total_profit:,.2f}", f"{profit_margin_pct:.2f}%",
              f"{order_count:,}", f"{unique_customers}", f"${avg_order_value:,.2f}",
              f"{df['product_name'].nunique()}", datetime.now().strftime('%Y-%m-%d')]
}
pd.DataFrame(summary_stats).to_csv(os.path.join(script_dir, 'summary_statistics.csv'), index=False)
print(f"   ✓ Exported summary statistics to: summary_statistics.csv")

print("\n" + "=" * 60)
print("✅ ANALYSIS COMPLETE!")
print("=" * 60)
print(f"""
📁 Output Files Generated:
   • sales_by_region.png
   • sales_by_category.png
   • monthly_sales_trend.png
   • sales_vs_profit_by_category.png
   • top_10_products.png
   • sales_by_segment.png
   • quarterly_sales_comparison.png
   • profit_margin_heatmap.png
   • sales_by_day_of_week.png
   • discount_impact_scatter.png
   • summary_statistics.csv
   • cleaned_sales_data.csv (for Power BI)
   • cleaned_sales_data.xlsx (for Power BI)
""")
