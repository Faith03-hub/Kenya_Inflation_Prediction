# src/data_loader.py - Universal Data Loader for Any File

import pandas as pd
import numpy as np
import re
from datetime import datetime

class UniversalDataLoader:
    """Automatically detects and loads inflation data from any file format"""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.xls']
        
    def detect_file_type(self, file):
        """Detect file type and return appropriate reader"""
        if file.name.endswith('.csv'):
            return self._read_csv
        elif file.name.endswith(('.xlsx', '.xls')):
            return self._read_excel
        else:
            raise ValueError(f"Unsupported file format: {file.name}")
    
    def _read_csv(self, file):
        """Read CSV with multiple encoding attempts"""
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                file.seek(0)
                return pd.read_csv(file, encoding=encoding)
            except UnicodeDecodeError:
                continue
            except Exception:
                continue
        
        # Last resort
        file.seek(0)
        return pd.read_csv(file, engine='python')
    
    def _read_excel(self, file):
        """Read Excel with automatic sheet detection"""
        # Try to find sheets that might contain inflation data
        excel_file = pd.ExcelFile(file)
        sheet_names = excel_file.sheet_names
        
        # Priority sheets (KNBS Economic Survey)
        priority_sheets = [
            'Table 3.16', 'Table 3.15a', 'Table 3.22',
            'Data', 'Inflation', 'CPI', 'Sheet1'
        ]
        
        for sheet in priority_sheets:
            if sheet in sheet_names:
                try:
                    file.seek(0)
                    return pd.read_excel(file, sheet_name=sheet)
                except:
                    continue
        
        # If no priority sheet found, try all sheets
        for sheet in sheet_names:
            try:
                file.seek(0)
                df = pd.read_excel(file, sheet_name=sheet)
                if len(df) > 0:
                    return df
            except:
                continue
        
        # Last resort - read first sheet
        file.seek(0)
        return pd.read_excel(file, sheet_name=0)
    
    def find_year_column(self, df):
        """Intelligently find the year column"""
        # Patterns to look for
        year_patterns = [
            r'year', r'yr', r'date', r'period', r'time',
            r'fiscal', r'calendar', r'annual'
        ]
        
        # Check column names
        for col in df.columns:
            col_lower = str(col).lower()
            for pattern in year_patterns:
                if re.search(pattern, col_lower):
                    return col
        
        # Check if any column contains only years (4-digit numbers)
        for col in df.columns:
            sample = df[col].dropna().head(10)
            if len(sample) > 0:
                # Check if values look like years (1900-2100 range)
                try:
                    numeric_sample = pd.to_numeric(sample, errors='coerce').dropna()
                    if len(numeric_sample) > 0:
                        if all(1900 <= x <= 2100 for x in numeric_sample):
                            return col
                except:
                    continue
        
        return None
    
    def find_inflation_column(self, df):
        """Intelligently find the inflation rate column"""
        # Patterns to look for
        inflation_patterns = [
            r'inflation', r'cpi', r'price', r'consumer',
            r'rate', r'percent', r'%', r'pct',
            r'change', r'growth', r'index'
        ]
        
        # Check column names
        for col in df.columns:
            col_lower = str(col).lower()
            for pattern in inflation_patterns:
                if re.search(pattern, col_lower):
                    return col
        
        # Look for columns with typical inflation values (0-100 range)
        for col in df.columns:
            sample = df[col].dropna().head(10)
            if len(sample) > 0:
                try:
                    numeric_sample = pd.to_numeric(sample, errors='coerce').dropna()
                    if len(numeric_sample) > 0:
                        # Inflation rates are typically between -10 and 100
                        if all(-10 <= x <= 100 for x in numeric_sample):
                            # Additional check: values shouldn't be huge (not GDP or population)
                            if max(abs(numeric_sample)) < 1000:
                                return col
                except:
                    continue
        
        return None
    
    def extract_knbs_data(self, df):
        """Special handling for KNBS format"""
        try:
            # Look for KNBS patterns
            for idx, row in df.iterrows():
                row_text = ' '.join([str(x).lower() for x in row.values if pd.notna(x)])
                
                # Check for Overall Inflation row
                if 'overall inflation' in row_text:
                    # Get the next row which has the values
                    if idx + 1 < len(df):
                        values = df.iloc[idx + 1]
                        years = [2020, 2021, 2022, 2023, 2024]
                        inflation_values = []
                        
                        for i, year in enumerate(years):
                            # Find the column with the value
                            for col in df.columns:
                                try:
                                    val = pd.to_numeric(values[col], errors='coerce')
                                    if pd.notna(val):
                                        inflation_values.append(val)
                                        break
                                except:
                                    continue
                        
                        if len(inflation_values) == 5:
                            return pd.DataFrame({
                                'Year': years,
                                'Inflation_Rate': inflation_values
                            })
            
            # Check for KNBS Table 3.22 format (CPI data)
            for idx, row in df.iterrows():
                if 'annual average' in str(row.iloc[0]).lower():
                    if idx < len(df):
                        cpi_row = df.iloc[idx]
                        years = [2020, 2021, 2022, 2023, 2024]
                        cpi_values = []
                        
                        for col in range(1, 6):
                            try:
                                val = pd.to_numeric(cpi_row.iloc[col], errors='coerce')
                                if pd.notna(val):
                                    cpi_values.append(val)
                            except:
                                continue
                        
                        if len(cpi_values) == 5:
                            # Calculate inflation from CPI
                            inflation_values = [cpi_values[0]]
                            for i in range(1, len(cpi_values)):
                                if cpi_values[i-1] > 0:
                                    infl = ((cpi_values[i] - cpi_values[i-1]) / cpi_values[i-1]) * 100
                                    inflation_values.append(round(infl, 2))
                            
                            return pd.DataFrame({
                                'Year': years,
                                'Inflation_Rate': inflation_values
                            })
            
        except Exception as e:
            pass
        
        return None
    
    def clean_data(self, df, year_col, infl_col):
        """Clean and standardize the data"""
        result = pd.DataFrame()
        
        # Handle year column
        result['Year'] = pd.to_numeric(df[year_col], errors='coerce')
        
        # Handle inflation column
        result['Inflation_Rate'] = pd.to_numeric(df[infl_col], errors='coerce')
        
        # Remove invalid rows
        result = result.dropna()
        
        # Convert year to datetime
        result['Year'] = pd.to_datetime(result['Year'], format='%Y', errors='coerce')
        result = result.dropna()
        
        # Remove duplicates
        result = result.drop_duplicates(subset=['Year'])
        
        # Sort by year
        result = result.sort_values('Year')
        
        return result
    
    def load_file(self, file):
        """Main method to load any file and return standardized dataframe"""
        
        # Step 1: Read the file
        reader = self.detect_file_type(file)
        df = reader(file)
        
        if df is None or df.empty:
            return None, "File is empty or could not be read"
        
        # Step 2: Try KNBS special extraction first
        knbs_data = self.extract_knbs_data(df)
        if knbs_data is not None and not knbs_data.empty:
            return knbs_data, "KNBS Economic Survey (auto-detected)"
        
        # Step 3: Find year and inflation columns
        year_col = self.find_year_column(df)
        infl_col = self.find_inflation_column(df)
        
        if year_col is None:
            return None, "Could not find a year/date column. Please ensure your file has a column with years (e.g., 'Year', 'Date', 'Period')"
        
        if infl_col is None:
            return None, "Could not find an inflation column. Please ensure your file has a column with inflation rates (e.g., 'Inflation', 'CPI', 'Rate')"
        
        # Step 4: Clean and standardize
        try:
            cleaned_df = self.clean_data(df, year_col, infl_col)
            
            if cleaned_df.empty:
                return None, "No valid data found after cleaning"
            
            # Step 5: Validate data range
            if len(cleaned_df) < 3:
                return None, f"Only {len(cleaned_df)} years of data found. Need at least 3 years for forecasting."
            
            return cleaned_df, f"Successfully loaded {len(cleaned_df)} years of data ({cleaned_df['Year'].dt.year.min()}-{cleaned_df['Year'].dt.year.max()})"
            
        except Exception as e:
            return None, f"Error processing data: {str(e)}"

# Create a singleton instance
loader = UniversalDataLoader()

def load_any_file(file):
    """Convenience function to load any file"""
    return loader.load_file(file)
