# Power BI Dashboard Guide
## Retail Sales Performance Analysis

This guide provides step-by-step instructions for creating an interactive dashboard in Power BI Desktop.

---

## 📥 Step 1: Import Data

### Method 1: Import CSV
1. Open **Power BI Desktop**
2. Click **Home → Get Data → Text/CSV**
3. Navigate to `3_Power_BI/cleaned_sales_data.csv`
4. Click **Load**

### Method 2: Import Excel
1. Click **Home → Get Data → Excel Workbook**
2. Select `3_Power_BI/cleaned_sales_data.xlsx`
3. Select the sheet and click **Load**

---

## 📊 Step 2: Verify Data Types

Go to **Model View** and verify these data types:

| Column | Type | Format |
|--------|------|--------|
| order_date | Date | Short Date |
| ship_date | Date | Short Date |
| sales | Decimal Number | Currency |
| profit | Decimal Number | Currency |
| quantity | Whole Number | - |
| discount | Decimal Number | Percentage |
| shipping_cost | Decimal Number | Currency |

---

## 📏 Step 3: Create DAX Measures

Click **Modeling → New Measure** and create each of the following:

### Key Performance Indicators (KPIs)

```dax
Total Sales = SUM(cleaned_sales_data[sales])
```

```dax
Total Profit = SUM(cleaned_sales_data[profit])
```

```dax
Profit Margin % = 
DIVIDE([Total Profit], [Total Sales], 0) * 100
```

```dax
Total Orders = DISTINCTCOUNT(cleaned_sales_data[order_id])
```

```dax
Total Quantity = SUM(cleaned_sales_data[quantity])
```

```dax
Average Order Value = AVERAGE(cleaned_sales_data[sales])
```

```dax
Total Customers = DISTINCTCOUNT(cleaned_sales_data[customer_id])
```

### Year-over-Year Comparison

```dax
Sales YoY % = 
VAR CurrentYearSales = CALCULATE([Total Sales], YEAR(cleaned_sales_data[order_date]) = 2024)
VAR PreviousYearSales = CALCULATE([Total Sales], YEAR(cleaned_sales_data[order_date]) = 2023)
RETURN
DIVIDE(CurrentYearSales - PreviousYearSales, PreviousYearSales, 0) * 100
```

```dax
Profit YoY % = 
VAR CurrentYearProfit = CALCULATE([Total Profit], YEAR(cleaned_sales_data[order_date]) = 2024)
VAR PreviousYearProfit = CALCULATE([Total Profit], YEAR(cleaned_sales_data[order_date]) = 2023)
RETURN
DIVIDE(CurrentYearProfit - PreviousYearProfit, PreviousYearProfit, 0) * 100
```

### Advanced Metrics

```dax
Avg Shipping Days = AVERAGE(cleaned_sales_data[shipping_days])
```

```dax
Discount Amount = 
SUMX(
    cleaned_sales_data,
    cleaned_sales_data[sales] * cleaned_sales_data[discount] / (1 - cleaned_sales_data[discount])
)
```

```dax
Orders with Discount = 
CALCULATE(
    [Total Orders],
    cleaned_sales_data[discount] > 0
)
```

```dax
Orders without Discount = 
CALCULATE(
    [Total Orders],
    cleaned_sales_data[discount] = 0
)
```

---

## 🎨 Step 4: Create Dashboard Layout

### Page 1: Executive Overview

**Layout Suggestion:**

```
┌─────────────────────────────────────────────────────────────┐
│                    RETAIL SALES DASHBOARD                    │
│                     [Date Range Slicer]                      │
├────────────┬────────────┬────────────┬────────────┬─────────┤
│ Total Sales│Total Profit│Profit %    │Orders      │Customers│
│  KPI Card  │  KPI Card  │  KPI Card  │  KPI Card  │ KPI Card│
├────────────┴────────────┴────────────┴────────────┴─────────┤
│                                                              │
│   [Sales by Region - Bar Chart]  [Sales by Category - Pie]  │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│              [Monthly Sales Trend - Line Chart]              │
│                                                              │
├────────────────────────────┬─────────────────────────────────┤
│ [Top Products - Table]     │  [Sales by Segment - Donut]    │
└────────────────────────────┴─────────────────────────────────┘
```

### Visual Specifications

#### 1. KPI Cards (Top Row)
- **Visual Type:** Card
- **Fields:** 
  - Total Sales (format as $ currency)
  - Total Profit
  - Profit Margin %
  - Total Orders
  - Total Customers

#### 2. Sales by Region (Clustered Bar Chart)
- **Visual Type:** Clustered Bar Chart
- **Axis:** region
- **Values:** Total Sales, Total Profit
- **Colors:** Blue for Sales, Orange for Profit

#### 3. Sales by Category (Pie Chart)
- **Visual Type:** Pie Chart
- **Legend:** category
- **Values:** Total Sales
- **Show Labels:** Yes (with %)

#### 4. Monthly Sales Trend (Line Chart)
- **Visual Type:** Line Chart
- **X-Axis:** year_month (Date Hierarchy)
- **Y-Axis:** Total Sales
- **Secondary Y-Axis:** Total Profit
- **Add Data Labels:** Yes

#### 5. Top 10 Products (Table)
- **Visual Type:** Table
- **Columns:** product_name, Total Sales, Total Profit, Total Quantity
- **Filter:** Top 10 by Total Sales

#### 6. Sales by Segment (Donut Chart)
- **Visual Type:** Donut Chart
- **Legend:** segment
- **Values:** Total Sales

---

### Page 2: Detailed Analysis

**Layout Suggestion:**

```
┌─────────────────────────────────────────────────────────────┐
│                    DETAILED ANALYSIS                         │
│  [Region Slicer] [Category Slicer] [Segment Slicer]         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   [Sales by Sub-Category - Horizontal Bar Chart]            │
│                                                              │
├────────────────────────────┬─────────────────────────────────┤
│ [Quarterly Comparison]     │  [Profit Heatmap by Region]    │
│    Clustered Column        │     Matrix Visual              │
├────────────────────────────┴─────────────────────────────────┤
│                                                              │
│              [Discount Impact - Scatter Chart]               │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│ [Top Customers - Table]    │  [Ship Mode Analysis - Treemap]│
└────────────────────────────┴─────────────────────────────────┘
```

---

## 🎨 Step 5: Design Best Practices

### Color Theme
Use a consistent color palette:
- **Primary Blue:** #2E86AB
- **Accent Purple:** #A23B72
- **Accent Orange:** #F18F01
- **Danger Red:** #C73E1D
- **Success Green:** #44AF69

### Formatting Guidelines
1. **Background:** Light gray (#F5F5F5) or white
2. **Title Font:** Segoe UI, 18pt, Bold
3. **Number Format:** 
   - Currency: $1,234.56
   - Percentage: 15.2%
   - Large numbers: 1.2M for millions
4. **Card Borders:** Subtle shadow effect

### Interactivity
1. **Slicers to Add:**
   - Date Range (Year/Quarter/Month)
   - Region (Dropdown)
   - Category (Dropdown)
   - Segment (Buttons)

2. **Cross-Filtering:**
   - Enable visual interactions between all charts
   - Clicking a region filters all other visuals

3. **Drill-Through:**
   - Create a drill-through page for customer details
   - Enable on product_name to see order history

---

## 📱 Step 6: Add Page Navigation

### Create Navigation Buttons
1. Insert → Buttons → Blank
2. Format → Action → Page Navigation
3. Set destination to target page
4. Style with icons or text

### Suggested Pages
- Page 1: Executive Overview
- Page 2: Detailed Analysis  
- Page 3: Customer Insights
- Page 4: Product Performance

---

## 🔄 Step 7: Refresh & Publish

### Data Refresh
1. Click **Home → Refresh**
2. Verify all visuals update correctly

### Save & Publish
1. Save as `Retail_Sales_Dashboard.pbix`
2. (Optional) Publish to Power BI Service:
   - Click **Home → Publish**
   - Select your workspace
   - Share the dashboard link

---

## 📋 Dashboard Checklist

Before presenting your dashboard:

- [ ] All KPI cards show correct values
- [ ] Charts have proper titles and labels
- [ ] Color scheme is consistent throughout
- [ ] Slicers filter all visuals correctly
- [ ] Numbers are formatted (currency, %)
- [ ] Mobile layout is optimized
- [ ] All pages have navigation buttons
- [ ] Tooltips show additional context
- [ ] Dashboard loads in < 5 seconds

---

## 📊 Expected Insights from Dashboard

Based on the analysis, your dashboard should reveal:

| Insight | Value |
|---------|-------|
| Total Revenue | $1,425,656.14 |
| Total Profit | $216,875.57 |
| Profit Margin | 15.21% |
| Best Region | East |
| Top Category | Technology |
| Best Product | MacBook Pro 14" |
| Top Customer | Anita Iyer |

---

## 🚀 Next Steps

1. **Export to PDF:** File → Export → PDF
2. **Create PowerPoint:** File → Export → PowerPoint
3. **Schedule Refresh:** Set up automatic data refresh in Power BI Service
4. **Share with Stakeholders:** Create a workspace and share access

Good luck with your dashboard! 🎉
