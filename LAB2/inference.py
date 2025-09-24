import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime

def load_model_and_scaler():
    """Load the trained model and scaler"""
    try:
        # Load the best checkpoint model
        model = tf.keras.models.load_model('best_model_checkpoint.h5')
        scaler = joblib.load('scaler.pkl')
        print("✅ Model and scaler loaded successfully!")
        return model, scaler
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find model files. Please ensure the following files exist:")
        print("   - best_model_checkpoint.h5")
        print("   - scaler.pkl")
        print(f"   Error details: {e}")
        return None, None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def get_student_input():
    """Get student information from user input with validation"""
    print("\n" + "="*60)
    print("📚 ACADEMIC PERFORMANCE PREDICTION")
    print("="*60)
    print("Please enter the student's information:")
    print("-" * 40)
    
    while True:
        try:
            # Study hours input
            study_hours = float(input("📖 Study hours per week (0.5 - 15): "))
            if not (0.5 <= study_hours <= 15):
                print("⚠️  Please enter study hours between 0.5 and 15")
                continue
            break
        except ValueError:
            print("⚠️  Please enter a valid number for study hours")
    
    while True:
        try:
            # Attendance percentage input
            attendance = float(input("🎯 Attendance percentage (20 - 100): "))
            if not (20 <= attendance <= 100):
                print("⚠️  Please enter attendance percentage between 20 and 100")
                continue
            break
        except ValueError:
            print("⚠️  Please enter a valid number for attendance percentage")
    
    while True:
        try:
            # Previous GPA input
            previous_gpa = float(input("🎓 Previous GPA (0.0 - 4.0): "))
            if not (0.0 <= previous_gpa <= 4.0):
                print("⚠️  Please enter GPA between 0.0 and 4.0")
                continue
            break
        except ValueError:
            print("⚠️  Please enter a valid number for GPA")
    
    while True:
        try:
            # Age input
            age = int(input("👤 Age (18 - 25): "))
            if not (18 <= age <= 25):
                print("⚠️  Please enter age between 18 and 25")
                continue
            break
        except ValueError:
            print("⚠️  Please enter a valid integer for age")
    
    return study_hours, attendance, previous_gpa, age

def predict_performance(model, scaler, study_hours, attendance, previous_gpa, age):
    """Make prediction for student performance with attendance rule"""
    # Create input array
    student_data = np.array([[study_hours, attendance, previous_gpa, age]])
    
    # Scale the input using the fitted scaler
    student_data_scaled = scaler.transform(student_data)
    
    # Make prediction using the model
    model_probability = model.predict(student_data_scaled, verbose=0)[0][0]
    model_prediction = "PASS" if model_probability > 0.5 else "FAIL"
    
    # BUSINESS RULE: If attendance < 75%, automatic FAIL regardless of model prediction
    if attendance < 75.0:
        final_prediction = "FAIL"
        # Set probability to reflect the business rule override
        final_probability = min(model_probability, 0.3)  # Cap at 30% for low attendance
        rule_applied = True
    else:
        final_prediction = model_prediction
        final_probability = model_probability
        rule_applied = False
    
    return final_prediction, final_probability, model_prediction, model_probability, rule_applied

def display_prediction_result(study_hours, attendance, previous_gpa, age, prediction, probability, model_prediction, model_probability, rule_applied):
    """Display the prediction result in a formatted way"""
    print("\n" + "="*60)
    print("🔮 PREDICTION RESULTS")
    print("="*60)
    
    # Student summary
    print("👨‍🎓 Student Profile:")
    print(f"   📖 Study Hours/Week: {study_hours}")
    print(f"   🎯 Attendance: {attendance}%")
    print(f"   🎓 Previous GPA: {previous_gpa}")
    print(f"   👤 Age: {age}")
    
    print("\n" + "-"*60)
    
    # Show model prediction first
    print("🤖 AI Model Prediction:")
    if model_prediction == "PASS":
        print(f"   ✅ Model says: PASS (Confidence: {model_probability:.1%})")
    else:
        print(f"   ❌ Model says: FAIL (Confidence: {(1-model_probability):.1%})")
    
    print("\n" + "-"*30)
    
    # Final prediction result with business rule
    if rule_applied:
        print("⚠️  ATTENDANCE RULE APPLIED!")
        print(f"📋 Institution Policy: Attendance < 75% = Automatic FAIL")
        print(f"🎯 Your Attendance: {attendance}% (Below 75% threshold)")
        print("\n🏛️  FINAL DECISION: FAIL ❌")
        print("📢 Reason: Poor attendance - not eligible for promotion")
        print(f"💔 Despite model prediction of {model_prediction}, you cannot pass due to attendance policy")
        
        if model_prediction == "PASS":
            print(f"\n💡 Note: You study well (model predicted PASS with {model_probability:.1%} confidence)")
            print("   But regular attendance is mandatory for course completion!")
        
    else:
        print("✅ ATTENDANCE REQUIREMENT MET (≥75%)")
        print(f"🎯 Your Attendance: {attendance}%")
        print(f"\n🏛️  FINAL DECISION: {prediction}")
        
        if prediction == "PASS":
            print(f"🎉 Confidence: {probability:.1%}")
            
            if probability > 0.9:
                confidence_level = "Very High"
                emoji = "🌟"
            elif probability > 0.8:
                confidence_level = "High"
                emoji = "👍"
            elif probability > 0.7:
                confidence_level = "Good"
                emoji = "😊"
            else:
                confidence_level = "Moderate"
                emoji = "🤞"
                
            print(f"{emoji} Confidence Level: {confidence_level}")
            
        else:
            print(f"😞 Confidence: {(1-probability):.1%}")
            
            if probability < 0.1:
                confidence_level = "Very High"
                emoji = "⚠️"
            elif probability < 0.2:
                confidence_level = "High"
                emoji = "😟"
            elif probability < 0.3:
                confidence_level = "Good"
                emoji = "😕"
            else:
                confidence_level = "Moderate"
                emoji = "🤔"
                
            print(f"{emoji} Confidence Level: {confidence_level}")
    
    print("\n" + "-"*60)
    
    # Recommendations based on input and rule application
    print("💡 RECOMMENDATIONS:")
    
    if rule_applied:
        print("   🚨 URGENT: Improve attendance immediately!")
        print("   📈 Target: Achieve at least 75% attendance to be eligible")
        print("   ⏰ Action: Attend all remaining classes without fail")
        if study_hours >= 8:
            print("   👏 Good study habits detected - combine with better attendance!")
        else:
            print("   📚 Also consider increasing study hours alongside attendance")
    else:
        if attendance < 80:
            print("   📈 Good attendance but aim for 80%+ for better performance")
        elif attendance < 90:
            print("   📊 Great attendance, try to maintain or improve further")
        else:
            print("   ✅ Excellent attendance - keep it up!")
        
        if study_hours < 5:
            print("   📚 Consider increasing study hours for better performance")
        elif study_hours < 8:
            print("   👍 Good study habits, maintain consistency")
        else:
            print("   🌟 Excellent study dedication!")
        
        if previous_gpa < 2.5:
            print("   🎯 Focus on improving understanding of fundamental concepts")
        elif previous_gpa < 3.0:
            print("   📖 Good foundation, keep building on your knowledge")
        else:
            print("   🎓 Strong academic background!")
    
    # Institution policy reminder
    print("\n📋 INSTITUTION POLICY REMINDER:")
    print("   • Minimum 75% attendance required for course completion")
    print("   • No exceptions to attendance policy")
    print("   • AI model assists but policy rules are final")

def save_prediction_log(study_hours, attendance, previous_gpa, age, prediction, probability, model_prediction, model_probability, rule_applied):
    """Save prediction to a log file"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            'timestamp': timestamp,
            'study_hours': study_hours,
            'attendance': attendance,
            'previous_gpa': previous_gpa,
            'age': age,
            'model_prediction': model_prediction,
            'model_probability': model_probability,
            'final_prediction': prediction,
            'final_probability': probability,
            'attendance_rule_applied': rule_applied,
            'attendance_threshold_met': attendance >= 75.0
        }
        
        # Create or append to log file
        log_file = 'prediction_log.csv'
        if os.path.exists(log_file):
            df_log = pd.read_csv(log_file)
            df_log = pd.concat([df_log, pd.DataFrame([log_entry])], ignore_index=True)
        else:
            df_log = pd.DataFrame([log_entry])
        
        df_log.to_csv(log_file, index=False)
        print(f"📝 Prediction logged to {log_file}")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not save to log file: {e}")

def batch_predict_from_csv():
    """Predict for multiple students from a CSV file"""
    csv_file = input("\n📁 Enter CSV file path (or press Enter to skip): ").strip()
    
    if not csv_file:
        return
    
    try:
        df = pd.read_csv(csv_file)
        print(f"📊 Loaded {len(df)} students from CSV")
        
        # Check required columns
        required_cols = ['study_hours_per_week', 'attendance_percentage', 'previous_gpa', 'age']
        if not all(col in df.columns for col in required_cols):
            print(f"❌ CSV must contain columns: {required_cols}")
            return
        
        # Load model
        model, scaler = load_model_and_scaler()
        if model is None:
            return
        
        # Make predictions
        predictions = []
        probabilities = []
        model_predictions = []
        model_probabilities = []
        rules_applied = []
        
        for _, row in df.iterrows():
            final_pred, final_prob, model_pred, model_prob, rule_applied = predict_performance(
                model, scaler, 
                row['study_hours_per_week'], 
                row['attendance_percentage'], 
                row['previous_gpa'], 
                row['age']
            )
            predictions.append(final_pred)
            probabilities.append(final_prob)
            model_predictions.append(model_pred)
            model_probabilities.append(model_prob)
            rules_applied.append(rule_applied)
        
        # Add results to dataframe
        df['model_prediction'] = model_predictions
        df['model_probability'] = model_probabilities
        df['final_prediction'] = predictions
        df['final_probability'] = probabilities
        df['attendance_rule_applied'] = rules_applied
        df['attendance_threshold_met'] = df['attendance_percentage'] >= 75.0
        
        # Save results
        output_file = 'batch_predictions.csv'
        df.to_csv(output_file, index=False)
        
        print(f"✅ Batch predictions saved to {output_file}")
        
        # Detailed summary
        total_students = len(predictions)
        final_pass = sum(1 for p in predictions if p == 'PASS')
        final_fail = sum(1 for p in predictions if p == 'FAIL')
        model_pass = sum(1 for p in model_predictions if p == 'PASS')
        model_fail = sum(1 for p in model_predictions if p == 'FAIL')
        attendance_fails = sum(1 for r in rules_applied if r)
        
        print(f"\n📊 BATCH PREDICTION SUMMARY:")
        print(f"   👥 Total Students: {total_students}")
        print(f"   🤖 Model Predictions - PASS: {model_pass}, FAIL: {model_fail}")
        print(f"   🏛️  Final Results - PASS: {final_pass}, FAIL: {final_fail}")
        print(f"   ⚠️  Students failed due to attendance (<75%): {attendance_fails}")
        
        if attendance_fails > 0:
            print(f"   💡 Note: {attendance_fails} students had their model predictions overridden due to poor attendance")
        
    except Exception as e:
        print(f"❌ Error processing CSV: {e}")

def main():
    """Main function to run the prediction system"""
    print("🎓 Academic Performance Prediction System")
    print("=" * 50)
    
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    if model is None:
        return
    
    while True:
        print("\n🔧 Choose an option:")
        print("1. 👤 Single student prediction")
        print("2. 📁 Batch prediction from CSV")
        print("3. 🚪 Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            # Single prediction
            try:
                # Get input from user
                study_hours, attendance, previous_gpa, age = get_student_input()
                
                # Make prediction
                prediction, probability, model_prediction, model_probability, rule_applied = predict_performance(
                    model, scaler, study_hours, attendance, previous_gpa, age
                )
                
                # Display results
                display_prediction_result(
                    study_hours, attendance, previous_gpa, age, 
                    prediction, probability, model_prediction, model_probability, rule_applied
                )
                
                # Ask if user wants to save the prediction
                save_choice = input("\n💾 Save this prediction to log? (y/n): ").lower()
                if save_choice == 'y':
                    save_prediction_log(
                        study_hours, attendance, previous_gpa, age, 
                        prediction, probability, model_prediction, model_probability, rule_applied
                    )
                
            except KeyboardInterrupt:
                print("\n\n👋 Prediction cancelled by user")
            except Exception as e:
                print(f"\n❌ Error during prediction: {e}")
        
        elif choice == '2':
            # Batch prediction
            batch_predict_from_csv()
        
        elif choice == '3':
            print("\n👋 Thank you for using Academic Performance Prediction System!")
            break
        
        else:
            print("⚠️  Invalid choice. Please enter 1, 2, or 3.")
        
        # Ask if user wants to continue
        if choice in ['1', '2']:
            continue_choice = input("\n🔄 Make another prediction? (y/n): ").lower()
            if continue_choice != 'y':
                print("\n👋 Thank you for using Academic Performance Prediction System!")
                break

if __name__ == "__main__":
    main()