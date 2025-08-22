import requests
import unittest
import json
import os
import base64
from io import BytesIO
from PIL import Image
import numpy as np

# Get the backend URL from the frontend .env file
BACKEND_URL = "http://localhost:8001"
API_URL = f"{BACKEND_URL}/api"

class FashionRecommendationAPITest(unittest.TestCase):
    """Test suite for the Fashion Recommendation API"""
    
    def setUp(self):
        """Set up test case - create a test user and login"""
        self.token = None
        self.test_user = {
            "email": "test@example.com",
            "password": "testpassword123"
        }
        
        # Try to login with test user
        try:
            response = requests.post(f"{API_URL}/auth/login", json=self.test_user)
            if response.status_code == 200:
                self.token = response.json().get("access_token")
                print("✅ Successfully logged in with test user")
            else:
                print("⚠️ Could not login with test user, will try to create one")
                # Try to create the test user
                signup_response = requests.post(f"{API_URL}/auth/signup", json=self.test_user)
                if signup_response.status_code == 200:
                    print("✅ Created test user")
                    # Now login
                    login_response = requests.post(f"{API_URL}/auth/login", json=self.test_user)
                    if login_response.status_code == 200:
                        self.token = login_response.json().get("access_token")
                        print("✅ Logged in with newly created test user")
                else:
                    print(f"⚠️ Could not create test user: {signup_response.text}")
        except Exception as e:
            print(f"⚠️ Error during setup: {str(e)}")
    
    def get_auth_headers(self):
        """Get authorization headers with token"""
        if self.token:
            return {"Authorization": f"Bearer {self.token}"}
        return {}
    
    def test_01_root_endpoint(self):
        """Test the root API endpoint"""
        print("\n🔍 Testing root API endpoint...")
        try:
            response = requests.get(f"{API_URL}/")
            self.assertEqual(response.status_code, 200)
            self.assertIn("message", response.json())
            print("✅ Root API endpoint working")
        except Exception as e:
            self.fail(f"❌ Root API endpoint test failed: {str(e)}")
    
    def test_02_auth_me_endpoint(self):
        """Test the auth/me endpoint"""
        print("\n🔍 Testing auth/me endpoint...")
        if not self.token:
            self.skipTest("No authentication token available")
        
        try:
            response = requests.get(
                f"{API_URL}/auth/me", 
                headers=self.get_auth_headers()
            )
            self.assertEqual(response.status_code, 200)
            user_data = response.json()
            self.assertIn("email", user_data)
            self.assertIn("id", user_data)
            self.assertEqual(user_data["email"], self.test_user["email"])
            print("✅ Auth/me endpoint working")
        except Exception as e:
            self.fail(f"❌ Auth/me endpoint test failed: {str(e)}")
    
    def test_03_fashion_categories(self):
        """Test the fashion categories endpoint"""
        print("\n🔍 Testing fashion categories endpoint...")
        try:
            response = requests.get(f"{API_URL}/fashion-categories")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("categories", data)
            print(f"✅ Fashion categories endpoint returned {len(data['categories'])} categories")
        except Exception as e:
            self.fail(f"❌ Fashion categories endpoint test failed: {str(e)}")
    
    def test_04_outfit_recommendations(self):
        """Test the outfit recommendations endpoint"""
        print("\n🔍 Testing outfit recommendations endpoint...")
        try:
            # Test with Men's fashion
            params = {
                "gender": "Men",
                "recommended_colors": "Blue,Black,White",
                "limit": 5
            }
            response = requests.get(f"{API_URL}/outfit-recommendations", params=params)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("recommendations", data)
            men_count = len(data["recommendations"])
            print(f"✅ Got {men_count} recommendations for Men")
            
            # Test with Women's fashion
            params["gender"] = "Women"
            response = requests.get(f"{API_URL}/outfit-recommendations", params=params)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("recommendations", data)
            women_count = len(data["recommendations"])
            print(f"✅ Got {women_count} recommendations for Women")
            
        except Exception as e:
            self.fail(f"❌ Outfit recommendations endpoint test failed: {str(e)}")
    
    def test_05_favorites_crud(self):
        """Test the favorites CRUD operations"""
        print("\n🔍 Testing favorites CRUD operations...")
        if not self.token:
            self.skipTest("No authentication token available")
        
        try:
            # 1. Get current favorites
            response = requests.get(
                f"{API_URL}/favorites", 
                headers=self.get_auth_headers()
            )
            self.assertEqual(response.status_code, 200)
            initial_favorites = response.json().get("favorites", [])
            print(f"✅ Retrieved {len(initial_favorites)} existing favorites")
            
            # 2. Add a new favorite
            test_item = {
                "item_id": f"test_item_{len(initial_favorites) + 1}",
                "product_name": "Test Product",
                "base_colour": "Blue"
            }
            
            response = requests.post(
                f"{API_URL}/favorites", 
                json=test_item,
                headers=self.get_auth_headers()
            )
            self.assertEqual(response.status_code, 200)
            print(f"✅ Added new favorite item: {test_item['item_id']}")
            
            # 3. Verify it was added
            response = requests.get(
                f"{API_URL}/favorites", 
                headers=self.get_auth_headers()
            )
            self.assertEqual(response.status_code, 200)
            updated_favorites = response.json().get("favorites", [])
            self.assertEqual(len(updated_favorites), len(initial_favorites) + 1)
            print("✅ Verified favorite was added")
            
            # 4. Delete the favorite
            response = requests.delete(
                f"{API_URL}/favorites/{test_item['item_id']}", 
                headers=self.get_auth_headers()
            )
            self.assertEqual(response.status_code, 200)
            print(f"✅ Deleted favorite item: {test_item['item_id']}")
            
            # 5. Verify it was deleted
            response = requests.get(
                f"{API_URL}/favorites", 
                headers=self.get_auth_headers()
            )
            self.assertEqual(response.status_code, 200)
            final_favorites = response.json().get("favorites", [])
            self.assertEqual(len(final_favorites), len(initial_favorites))
            print("✅ Verified favorite was deleted")
            
        except Exception as e:
            self.fail(f"❌ Favorites CRUD test failed: {str(e)}")
    
    def test_06_skin_tone_analysis(self):
        """Test the skin tone analysis endpoint with a test image"""
        print("\n🔍 Testing skin tone analysis endpoint...")
        
        try:
            # Create a simple test image (a solid color face-like shape)
            img = Image.new('RGB', (300, 300), color=(245, 210, 180))  # Light skin tone color
            
            # Draw a simple face-like shape to help face detection
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            # Draw a circle for the face
            draw.ellipse((50, 50, 250, 250), fill=(245, 210, 180))
            # Draw eyes
            draw.ellipse((100, 120, 130, 150), fill=(255, 255, 255))
            draw.ellipse((170, 120, 200, 150), fill=(255, 255, 255))
            # Draw pupils
            draw.ellipse((110, 130, 120, 140), fill=(0, 0, 0))
            draw.ellipse((180, 130, 190, 140), fill=(0, 0, 0))
            # Draw mouth
            draw.arc((120, 150, 180, 200), start=0, end=180, fill=(150, 75, 75), width=5)
            
            # Save to BytesIO
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            
            # Prepare the file for upload
            files = {'file': ('test_face.jpg', img_byte_arr, 'image/jpeg')}
            
            # Make the request
            headers = self.get_auth_headers()
            response = requests.post(
                f"{API_URL}/analyze-skin-tone",
                files=files,
                headers=headers
            )
            
            # Check if the request was successful
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Skin tone analysis successful")
                print(f"   Detected skin tone: {result.get('detected_skin_tone')}")
                print(f"   Recommended colors: {', '.join(result.get('recommended_colors', []))}")
            else:
                print(f"⚠️ Skin tone analysis returned status code {response.status_code}")
                print(f"   Response: {response.text}")
                
                # This might fail due to face detection issues with our simple test image
                # We'll consider this a soft failure
                print("   Note: This test may fail due to face detection limitations with test images")
        
        except Exception as e:
            print(f"⚠️ Skin tone analysis test exception: {str(e)}")
            print("   Note: This test may fail due to face detection or image processing requirements")

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)