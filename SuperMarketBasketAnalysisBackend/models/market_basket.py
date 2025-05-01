import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import os
import tempfile
from collections import defaultdict
from itertools import combinations

class MarketBasketAnalysis:
    def __init__(self):
        """Initialize the Market Basket Analysis without PySpark"""
        self.df = None
        self.transactions = None
        self.frequent_itemsets = None
        self.association_rules = None
        self.min_support = 0.02
        self.min_confidence = 0.2

    def load_data(self, file_content):
        """Load data from uploaded file content"""
        # Save content to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name
        
        try:
            self.df = pd.read_excel(temp_path)
            return {"message": "Data loaded successfully", "rows": len(self.df)}
        except Exception as e:
            return {"error": str(e)}
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def clean_data(self):
        """Clean the data by removing nulls and duplicates"""
        if self.df is None:
            return {"error": "No data loaded"}
        
        try:
            initial_count = len(self.df)
            # Drop rows with missing CustomerID or Description
            self.df = self.df.dropna(subset=["CustomerID", "Description"])
            # Drop duplicates
            self.df = self.df.drop_duplicates()
            # Filter for positive quantities
            self.df = self.df[self.df["Quantity"] > 0]
            final_count = len(self.df)
            
            return {
                "message": "Data cleaned successfully",
                "initial_count": initial_count,
                "final_count": final_count,
                "removed_count": initial_count - final_count
            }
        except Exception as e:
            return {"error": str(e)}

    def prepare_transactions(self):
        """Group items by invoice to prepare for analysis"""
        if self.df is None:
            return {"error": "No data loaded"}
        
        try:
            # Group by InvoiceNo and collect unique items
            self.transactions = self.df.groupby('InvoiceNo')['Description'].apply(list).apply(lambda x: list(set(x))).reset_index()
            transaction_count = len(self.transactions)
            
            return {
                "message": "Transactions prepared successfully",
                "transaction_count": transaction_count
            }
        except Exception as e:
            return {"error": str(e)}

    def apply_fp_growth(self, min_support=0.02, min_confidence=0.2):
        """Apply market basket analysis to find frequent itemsets and association rules"""
        if self.transactions is None:
            return {"error": "Transactions not prepared"}
        
        try:
            self.min_support = min_support
            self.min_confidence = min_confidence
            
            # Simple implementation of frequent itemset mining
            item_counts = defaultdict(int)
            total_transactions = len(self.transactions)
            
            # Count individual items
            for _, row in self.transactions.iterrows():
                items = row['Description']
                for item in items:
                    item_counts[frozenset([item])] += 1
            
            # Filter by support threshold
            min_support_count = min_support * total_transactions
            frequent_1_itemsets = {item: count for item, count in item_counts.items() if count >= min_support_count}
            
            # Build L2 itemsets (pairs)
            pairs = {}
            for _, row in self.transactions.iterrows():
                items = row['Description']
                for pair in combinations([item for item in items if frozenset([item]) in frequent_1_itemsets], 2):
                    pair_set = frozenset(pair)
                    pairs[pair_set] = pairs.get(pair_set, 0) + 1
            
            # Filter pairs by support
            frequent_2_itemsets = {pair: count for pair, count in pairs.items() if count >= min_support_count}
            
            # Combine all frequent itemsets
            self.frequent_itemsets = []
            for item_set, count in frequent_1_itemsets.items():
                self.frequent_itemsets.append({
                    'items': list(item_set),
                    'freq': count,
                    'support': count / total_transactions
                })
            
            for item_set, count in frequent_2_itemsets.items():
                self.frequent_itemsets.append({
                    'items': list(item_set),
                    'freq': count,
                    'support': count / total_transactions
                })
            
            # Sort by frequency
            self.frequent_itemsets = sorted(self.frequent_itemsets, key=lambda x: x['freq'], reverse=True)
            
            # Generate simple association rules (for pairs only)
            self.association_rules = []
            for item_set, count in frequent_2_itemsets.items():
                items = list(item_set)
                # Generate rules for both directions
                for i in range(2):
                    antecedent = [items[i]]
                    consequent = [items[1-i]]
                    
                    # Calculate confidence
                    antecedent_support = item_counts[frozenset(antecedent)]
                    confidence = count / antecedent_support
                    
                    if confidence >= min_confidence:
                        self.association_rules.append({
                            'antecedent': antecedent,
                            'consequent': consequent,
                            'support': count / total_transactions,
                            'confidence': confidence
                        })
            
            # Sort rules by confidence
            self.association_rules = sorted(self.association_rules, key=lambda x: x['confidence'], reverse=True)
            
            # Prepare response
            sample_freq_items = self.frequent_itemsets[:20]
            sample_rules = self.association_rules[:20]
            
            return {
                "message": "Analysis applied successfully",
                "frequent_itemsets_count": len(self.frequent_itemsets),
                "rules_count": len(self.association_rules),
                "sample_frequent_itemsets": sample_freq_items,
                "sample_rules": sample_rules
            }
        except Exception as e:
            return {"error": str(e)}

    def visualize_frequent_itemsets(self):
        """Generate visualization of top frequent itemsets"""
        if self.frequent_itemsets is None:
            return {"error": "Model not trained"}
        
        try:
            plt.figure(figsize=(10, 6))
            
            # Get top 10 frequent itemsets
            top_itemsets = self.frequent_itemsets[:10]
            
            # Convert itemset lists to strings for display
            itemset_labels = []
            for item_dict in top_itemsets:
                items = item_dict['items']
                if len(items) < 3:
                    label = ', '.join(items)
                else:
                    label = ', '.join(items[:2]) + f'... ({len(items)} items)'
                itemset_labels.append(label)
                
            frequencies = [item_dict['freq'] for item_dict in top_itemsets]
            
            plt.barh(itemset_labels, frequencies, color='skyblue')
            plt.xlabel('Frequency')
            plt.ylabel('Itemsets')
            plt.title('Top 10 Frequent Itemsets')
            
            # Save plot to a base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            return {
                "message": "Visualization generated successfully",
                "visualization": f"data:image/png;base64,{image_base64}"
            }
        except Exception as e:
            return {"error": str(e)}

    def recommend_products(self, purchased_items, n_recommendations=3):
        """Recommend products based on purchased items"""
        if self.association_rules is None:
            return {"error": "Model not trained"}
        
        if not purchased_items:
            return {"error": "No purchased items provided"}
        
        try:
            # Filter rules where all purchased items are in the antecedent
            recommendations = []
            for rule in self.association_rules:
                if all(item in purchased_items for item in rule['antecedent']):
                    recommendations.append({
                        "items": rule['consequent'],
                        "confidence": rule['confidence'],
                        "support": rule['support']
                    })
            
            # Sort by confidence and limit to n_recommendations
            recommendations.sort(key=lambda x: x['confidence'], reverse=True)
            recommendations = recommendations[:n_recommendations]
            
            return {
                "message": "Recommendations generated successfully",
                "purchased_items": purchased_items,
                "recommendations": recommendations
            }
        except Exception as e:
            return {"error": str(e)}

    def shutdown(self):
        """Clean up method (no-op in this implementation)"""
        pass