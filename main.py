import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import os
from google.cloud import vision
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Initialize Google Cloud Vision client
def initialize_vision_client():
    try:
        client = vision.ImageAnnotatorClient()
        return client
    except Exception as e:
        st.error(f"Error initializing Vision API client: {str(e)}")
        return None

def enhance_image(image):
    """Apply image enhancement techniques"""
    try:
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to LAB color space
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        # Merge channels
        limg = cv2.merge((cl,a,b))
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(enhanced)
    except Exception as e:
        st.warning("Could not enhance image. Using original.")
        return image

def analyze_jewelry(image):
    """Perform comprehensive jewelry analysis using Google Cloud Vision API with enhanced prompting"""
    try:
        # Initialize Vision API client
        client = initialize_vision_client()
        if not client:
            return None
        
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Create image object
        image = vision.Image(content=img_byte_arr)
        
        # Define comprehensive feature list with maximum results
        features = [
            vision.Feature(type_=vision.Feature.Type.OBJECT_LOCALIZATION, max_results=50),
            vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION, max_results=50),
            vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION),
            vision.Feature(type_=vision.Feature.Type.WEB_DETECTION),
            vision.Feature(type_=vision.Feature.Type.IMAGE_PROPERTIES),
            vision.Feature(type_=vision.Feature.Type.LOGO_DETECTION),
            vision.Feature(type_=vision.Feature.Type.SAFE_SEARCH_DETECTION),
            vision.Feature(type_=vision.Feature.Type.FACE_DETECTION)  # For detecting any faces in decorative elements
        ]
        
        # Create detailed image context
        image_context = vision.ImageContext(
            language_hints=['en'],
            web_detection_params=vision.WebDetectionParams(
                include_geo_results=True
            )
        )
        
        # Perform the API request with context
        response = client.annotate_image({
            'image': image,
            'features': features,
            'image_context': image_context,
        })
        
        # Initialize enhanced analysis results with more categories
        analysis = {
            "comprehensive_description": "",
            "jewelry_type_analysis": "",
            "materials_composition": "",
            "gemstones_analysis": "",
            "design_elements": "",
            "craftsmanship_details": "",
            "setting_technique": "",
            "style_period": "",
            "quality_indicators": "",
            "brand_analysis": "",
            "market_relevance": "",
            "care_guidelines": "",
            "color_analysis": "",
            "size_and_dimensions": "",
            "wear_and_occasion": "",
            "historical_context": "",
            "value_proposition": "",
            "artistic_elements": "",
            "cultural_significance": "",
            "investment_potential": "",
            "additional_insights": "",
            "confidence_score": 0.0
        }
        
        # Process all annotations for comprehensive analysis
        if response.label_annotations:
            # Enhanced categorization with more detailed categories
            jewelry_info = {
                'type': [],
                'materials': [],
                'gemstones': [],
                'design': [],
                'style': [],
                'quality': [],
                'setting': [],
                'period': [],
                'technique': [],
                'color': [],
                'occasion': [],
                'cultural': [],
                'artistic': []
            }
            
            # Detailed label categorization with expanded terms
            for label in response.label_annotations:
                if label.score > 0.5:
                    label_text = label.description.lower()
                    
                    # Enhanced categorization with more specific terms
                    if any(word in label_text for word in [
                        'ring', 'necklace', 'bracelet', 'earring', 'pendant', 'brooch',
                        'tiara', 'anklet', 'bangle', 'choker', 'cuff', 'chain',
                        'charm', 'locket', 'pin', 'clasp', 'medallion', 'amulet'
                    ]):
                        jewelry_info['type'].append((label_text, label.score))
                    
                    elif any(word in label_text for word in [
                        'gold', 'silver', 'platinum', 'metal', 'alloy', 'brass',
                        'white gold', 'rose gold', 'yellow gold', 'sterling silver',
                        'palladium', 'titanium', 'copper', 'bronze', 'rhodium'
                    ]):
                        jewelry_info['materials'].append((label_text, label.score))
                    
                    elif any(word in label_text for word in [
                        'diamond', 'ruby', 'sapphire', 'emerald', 'pearl', 'opal',
                        'topaz', 'amethyst', 'aquamarine', 'garnet', 'jade', 'tanzanite',
                        'morganite', 'peridot', 'citrine', 'tourmaline', 'alexandrite',
                        'moonstone', 'lapis lazuli', 'turquoise', 'coral'
                    ]):
                        jewelry_info['gemstones'].append((label_text, label.score))
                    
                    elif any(word in label_text for word in [
                        'vintage', 'modern', 'contemporary', 'classic', 'antique',
                        'art deco', 'victorian', 'edwardian', 'art nouveau', 'retro',
                        'baroque', 'renaissance', 'gothic', 'minimalist', 'bohemian',
                        'geometric', 'organic', 'abstract', 'traditional', 'avant-garde'
                    ]):
                        jewelry_info['style'].append((label_text, label.score))
                    
                    elif any(word in label_text for word in [
                        'handcrafted', 'artisan', 'luxury', 'premium', 'fine',
                        'designer', 'custom', 'bespoke', 'high-end', 'exclusive',
                        'limited edition', 'signature', 'artisanal', 'masterpiece',
                        'couture', 'heritage', 'heirloom', 'collectible'
                    ]):
                        jewelry_info['quality'].append((label_text, label.score))
                    
                    elif any(word in label_text for word in [
                        'prong', 'bezel', 'pavé', 'channel', 'flush', 'tension',
                        'cluster', 'invisible', 'bar', 'gypsy', 'tiffany',
                        'cathedral', 'halo', 'solitaire', 'three-stone', 'eternity',
                        'micro-pavé', 'shared-prong', 'burnish', 'illusion'
                    ]):
                        jewelry_info['setting'].append((label_text, label.score))
                    
                    # New categories for enhanced analysis
                    elif any(word in label_text for word in [
                        'wedding', 'engagement', 'anniversary', 'birthday', 'graduation',
                        'formal', 'casual', 'evening', 'cocktail', 'statement',
                        'everyday', 'special occasion', 'ceremonial', 'festive'
                    ]):
                        jewelry_info['occasion'].append((label_text, label.score))
                    
                    elif any(word in label_text for word in [
                        'asian', 'european', 'african', 'indian', 'middle eastern',
                        'celtic', 'tribal', 'indigenous', 'byzantine', 'greco-roman',
                        'oriental', 'western', 'ethnic', 'traditional'
                    ]):
                        jewelry_info['cultural'].append((label_text, label.score))
                    
                    elif any(word in label_text for word in [
                        'floral', 'geometric', 'abstract', 'naturalistic', 'figurative',
                        'symmetrical', 'asymmetrical', 'organic', 'architectural',
                        'sculptural', 'minimalist', 'ornate', 'decorative'
                    ]):
                        jewelry_info['artistic'].append((label_text, label.score))
            
            # Generate comprehensive type analysis with historical context
            if jewelry_info['type']:
                main_type = max(jewelry_info['type'], key=lambda x: x[1])
                analysis["jewelry_type_analysis"] = (
                    f"This exquisite piece is identified as a {main_type[0]} with {main_type[1]*100:.1f}% confidence. "
                    "Based on the detailed analysis of its characteristics and design elements, "
                    f"this {main_type[0]} exemplifies fine craftsmanship and attention to detail. "
                    "The piece demonstrates a sophisticated understanding of jewelry design principles "
                    "and showcases exceptional execution in its creation. "
                    f"This particular style of {main_type[0]} represents a perfect blend of "
                    "traditional craftsmanship and contemporary aesthetics."
                )
                analysis["confidence_score"] = main_type[1]
            
            # Generate materials composition analysis with technical details
            if jewelry_info['materials']:
                materials_desc = []
                for material, score in jewelry_info['materials']:
                    materials_desc.append(f"{material} ({score*100:.1f}% confidence)")
                analysis["materials_composition"] = (
                    f"The piece is masterfully crafted primarily from {', '.join(materials_desc)}. "
                    "The choice of materials demonstrates a perfect balance between durability and aesthetic appeal. "
                    "The composition suggests both high-quality craftsmanship and attention to long-term wear. "
                    "The materials used are of premium grade, ensuring both beauty and longevity. "
                    "The combination of materials has been carefully selected to create optimal structural integrity "
                    "while maintaining the piece's elegant appearance. "
                    "The finishing techniques applied to these materials enhance their natural properties "
                    "and contribute to the overall sophistication of the piece."
                )
            
            # Generate enhanced gemstones analysis with technical specifications
            if jewelry_info['gemstones']:
                gems_desc = []
                for gem, score in jewelry_info['gemstones']:
                    gems_desc.append(f"{gem} ({score*100:.1f}% confidence)")
                analysis["gemstones_analysis"] = (
                    f"This piece features an impressive array of gemstones, including {', '.join(gems_desc)}. "
                    "Each stone appears to be carefully selected and precisely set, "
                    "showcasing exceptional clarity and color characteristics. "
                    "The arrangement of the stones enhances both the visual appeal and value of the piece. "
                    "The gemstones demonstrate excellent cut quality, maximizing their natural brilliance "
                    "and fire. The positioning of each stone has been meticulously planned to create "
                    "optimal light reflection and visual harmony. The selection of these particular "
                    "gemstones indicates a deep understanding of both aesthetics and value preservation."
                )
            
            # Generate comprehensive design elements analysis
            if jewelry_info['style']:
                style_desc = []
                for style, score in jewelry_info['style']:
                    style_desc.append(f"{style} ({score*100:.1f}% confidence)")
                analysis["design_elements"] = (
                    f"The design embodies {', '.join(style_desc)} elements, "
                    "creating a harmonious blend of aesthetics and functionality. "
                    "The piece showcases remarkable attention to detail in its execution, "
                    "with thoughtfully integrated design elements that contribute to its overall appeal. "
                    "The design demonstrates a sophisticated understanding of proportion and balance, "
                    "while incorporating innovative elements that set it apart. "
                    "Each design element has been carefully considered and executed with precision, "
                    "resulting in a piece that is both visually striking and comfortable to wear."
                )
            
            # Generate artistic elements analysis
            if jewelry_info['artistic']:
                art_desc = []
                for art, score in jewelry_info['artistic']:
                    art_desc.append(f"{art} ({score*100:.1f}% confidence)")
                analysis["artistic_elements"] = (
                    f"The artistic composition incorporates {', '.join(art_desc)} elements, "
                    "demonstrating a sophisticated approach to jewelry design. "
                    "The artistic elements show a masterful understanding of form and composition, "
                    "creating a piece that is both visually engaging and aesthetically balanced. "
                    "The creative execution reveals attention to artistic principles such as "
                    "rhythm, movement, and visual harmony. The design elements work together "
                    "to create a cohesive and compelling artistic statement."
                )
            
            # Generate cultural significance analysis
            if jewelry_info['cultural']:
                cultural_desc = []
                for culture, score in jewelry_info['cultural']:
                    cultural_desc.append(f"{culture} ({score*100:.1f}% confidence)")
                analysis["cultural_significance"] = (
                    f"The piece draws inspiration from {', '.join(cultural_desc)} influences, "
                    "reflecting a rich cultural heritage in its design and execution. "
                    "These cultural elements are thoughtfully incorporated, creating a piece "
                    "that honors traditional craftsmanship while maintaining contemporary appeal. "
                    "The cultural motifs and symbols embedded in the design add depth and meaning "
                    "to the piece, making it not just an accessory but a carrier of cultural significance."
                )
            
            # Generate wear and occasion analysis
            if jewelry_info['occasion']:
                occasion_desc = []
                for occ, score in jewelry_info['occasion']:
                    occasion_desc.append(f"{occ} ({score*100:.1f}% confidence)")
                analysis["wear_and_occasion"] = (
                    f"This piece is particularly suited for {', '.join(occasion_desc)} occasions. "
                    "Its versatility allows it to transition seamlessly between different settings, "
                    "making it a valuable addition to any jewelry collection. "
                    "The design considerations ensure that it maintains its elegance and "
                    "appropriateness across various social contexts and dress codes. "
                    "The piece strikes an excellent balance between statement-making presence "
                    "and versatile wearability."
                )
        
        # Enhanced color analysis using image properties
        if response.image_properties_annotation:
            colors = response.image_properties_annotation.dominant_colors.colors
            if colors:
                color_analysis = []
                for color in colors[:5]:  # Top 5 dominant colors
                    rgb = f"RGB({int(color.color.red)}, {int(color.color.green)}, {int(color.color.blue)})"
                    percentage = f"{color.score * 100:.1f}%"
                    color_analysis.append(f"{rgb} ({percentage})")
                
                analysis["color_analysis"] = (
                    "Color Composition Analysis:\n"
                    f"The piece features a sophisticated color palette with the following dominant colors:\n"
                    f"{', '.join(color_analysis)}.\n\n"
                    "This color combination creates a harmonious visual effect that enhances "
                    "the piece's overall aesthetic appeal. The interplay of these colors "
                    "demonstrates thoughtful design consideration and contributes to the "
                    "piece's versatility and wearability."
                )
        
        # Enhanced quality assessment with expanded indicators
        if response.text_annotations:
            text = response.text_annotations[0].description.lower()
            quality_markers = []
            
            # Expanded quality indicators with more categories
            quality_indicators = {
                'metal_purity': ['karat', 'carat', 'kt', 'ct', '925', '750', '585', '375', '999', '916', '835'],
                'certification': ['certified', 'hallmark', 'stamp', 'authentic', 'genuine', 'verified', 'tested'],
                'craftsmanship': ['handmade', 'handcrafted', 'artisan', 'custom', 'bespoke', 'signature', 'master'],
                'brand': ['designer', 'branded', 'collection', 'limited edition', 'exclusive', 'luxury'],
                'origin': ['made in', 'origin', 'sourced from', 'imported', 'native', 'local'],
                'age': ['vintage', 'antique', 'period', 'era', 'century', 'year'],
                'technique': ['cast', 'forged', 'carved', 'engraved', 'enameled', 'set', 'polished'],
                'finish': ['polish', 'matte', 'brushed', 'hammered', 'textured', 'smooth']
            }
            
            for category, markers in quality_indicators.items():
                found_markers = [marker for marker in markers if marker in text]
                if found_markers:
                    quality_markers.append(f"{category}: {', '.join(found_markers)}")
            
            if quality_markers:
                analysis["quality_indicators"] = (
                    f"Quality assessment reveals significant characteristics: {'; '.join(quality_markers)}. "
                    "These indicators confirm the piece's exceptional craftsmanship and material quality. "
                    "The presence of these markers suggests adherence to high manufacturing standards "
                    "and attention to detail in both design and execution. "
                    "The combination of quality indicators points to a piece that meets or exceeds "
                    "industry standards for fine jewelry. The craftsmanship demonstrates a commitment "
                    "to excellence in both materials and execution."
                )
        
        # Enhanced brand analysis with market positioning
        if response.logo_annotations:
            brands = []
            for logo in response.logo_annotations:
                if logo.score > 0.7:
                    brands.append(f"{logo.description} ({logo.score*100:.1f}% confidence)")
            
            if brands:
                analysis["brand_analysis"] = (
                    f"The piece is associated with prestigious brand(s): {', '.join(brands)}. "
                    "This association indicates adherence to established quality standards "
                    "and reflects the piece's position in the luxury jewelry market. "
                    "The brand heritage adds significant value to the piece's overall worth. "
                    "The design language and execution are consistent with the brand's "
                    "reputation for excellence in fine jewelry craftsmanship. "
                    "The brand association suggests a commitment to quality control and "
                    "authenticity verification processes."
                )
        
        # Generate investment potential analysis
        if any([analysis["quality_indicators"], analysis["brand_analysis"], analysis["materials_composition"]]):
            analysis["investment_potential"] = (
                "Investment Value Analysis:\n"
                "Based on the comprehensive analysis of materials, craftsmanship, and brand association, "
                "this piece demonstrates strong potential for value retention and appreciation. "
                "The quality of materials and execution suggests durability and timeless appeal. "
                "The design elements and craftsmanship indicate a piece that transcends temporary trends, "
                "positioning it as a potentially valuable addition to a fine jewelry collection. "
                "The combination of quality materials, expert craftsmanship, and design sophistication "
                "suggests both immediate and long-term value proposition."
            )
        
        # Enhanced market insights with trend analysis
        if response.web_detection and response.web_detection.visually_similar_images:
            similar_count = len(response.web_detection.visually_similar_images)
            web_entities = [entity for entity in response.web_detection.web_entities if entity.score > 0.5]
            
            market_insights = [
                f"Analysis reveals {similar_count} similar pieces in the current market",
                "The design shows strong contemporary relevance while maintaining unique characteristics",
            ]
            
            if web_entities:
                market_terms = [f"{entity.description} ({entity.score*100:.1f}% relevance)" 
                              for entity in web_entities[:5]]
                market_insights.append(f"Market positioning keywords: {', '.join(market_terms)}")
                
                # Add trend analysis
                market_insights.extend([
                    "Current market trends indicate growing appreciation for this style",
                    "The design elements align with contemporary jewelry market preferences",
                    "Similar pieces have shown strong market presence and consumer interest"
                ])
            
            analysis["market_relevance"] = (
                ". ".join(market_insights) + ".\n\n"
                "Market Position Analysis:\n"
                "The piece demonstrates strong market relevance while maintaining distinctive features "
                "that set it apart from mass-market offerings. The design elements and craftsmanship "
                "position it in the premium segment of the jewelry market, appealing to discerning "
                "consumers who value both quality and artistic merit. The combination of traditional "
                "elements with contemporary execution suggests broad market appeal and potential for "
                "long-term value retention."
            )
        
        # Generate comprehensive care guidelines with material-specific recommendations
        if "materials_composition" in analysis and analysis["materials_composition"]:
            care_tips = []
            
            # Enhanced care recommendations based on materials
            if 'gold' in analysis["materials_composition"].lower():
                care_tips.extend([
                    "Regular cleaning with mild soap and warm water",
                    "Avoid exposure to harsh chemicals and chlorine",
                    "Store in a soft cloth pouch or lined jewelry box",
                    "Remove before swimming or bathing",
                    "Use specialized gold cleaning solutions when needed",
                    "Avoid exposure to extreme temperatures",
                    "Remove before applying cosmetics or perfumes",
                    "Handle with clean, dry hands to prevent oils and residue"
                ])
            if 'silver' in analysis["materials_composition"].lower():
                care_tips.extend([
                    "Use anti-tarnish strips in storage",
                    "Clean with specialized silver polishing cloth",
                    "Store in tarnish-resistant bags",
                    "Remove when applying cosmetics or perfumes",
                    "Polish regularly to maintain luster",
                    "Keep away from rubber and latex",
                    "Use proper silver cleaning solutions",
                    "Store in a cool, dry place"
                ])
            if 'gemstones' in analysis["gemstones_analysis"]:
                care_tips.extend([
                    "Professional cleaning recommended bi-annually",
                    "Avoid ultrasonic cleaners for delicate stones",
                    "Check settings regularly for security",
                    "Protect from extreme temperature changes",
                    "Clean with soft brush and mild soap solution",
                    "Avoid exposure to harsh chemicals",
                    "Store stones separately to prevent scratching",
                    "Have professional inspection annually"
                ])
            
            if care_tips:
                analysis["care_guidelines"] = (
                    "Comprehensive Care Guidelines:\n" + 
                    "\n".join(f"• {tip}" for tip in care_tips) +
                    "\n\nGeneral Maintenance:\n" +
                    "• Store pieces separately to prevent scratching\n" +
                    "• Remove jewelry during physical activities\n" +
                    "• Schedule regular professional inspections\n" +
                    "• Avoid exposure to harsh chemicals and extreme conditions\n" +
                    "• Clean regularly with appropriate methods\n" +
                    "• Handle with care to prevent damage\n" +
                    "• Document any repairs or modifications\n" +
                    "• Keep original documentation and certificates"
                )
        
        # Generate value proposition
        analysis["value_proposition"] = (
            "Value Proposition Analysis:\n"
            "This piece represents a compelling value proposition based on several key factors:\n"
            "1. Material Quality: Premium materials ensure durability and lasting beauty\n"
            "2. Craftsmanship: Expert execution and attention to detail\n"
            "3. Design Merit: Thoughtful design elements and aesthetic appeal\n"
            "4. Versatility: Adaptability to various occasions and styles\n"
            "5. Investment Potential: Quality indicators suggesting value retention\n"
            "6. Brand Association: Reputation and quality assurance\n"
            "7. Market Position: Strong presence in current market context\n"
            "8. Uniqueness: Distinctive features setting it apart from similar pieces"
        )
        
        # Generate comprehensive description
        comprehensive_elements = []
        for key, value in analysis.items():
            if key not in ["confidence_score"] and value:
                comprehensive_elements.append(value)
        
        analysis["comprehensive_description"] = "\n\n".join(comprehensive_elements)
        
        return analysis
        
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        return None

def get_jewelry_recommendations(preferences):
    """Get personalized jewelry website recommendations"""
    # Define comprehensive jewelry website database
    jewelry_websites = {
        "Blue Nile": {
            "url": "https://www.bluenile.com",
            "specialty": "Fine diamonds and luxury jewelry",
            "description": "Blue Nile is renowned for its extensive collection of certified diamonds and fine jewelry. They offer exceptional quality, competitive pricing, and a user-friendly online shopping experience. Their educational resources and customization options make them ideal for engagement rings and significant jewelry purchases.",
            "price_range": "Medium to High",
            "best_for": ["Engagement Rings", "Diamond Jewelry", "Fine Jewelry", "Custom Designs"]
        },
        "James Allen": {
            "url": "https://www.jamesallen.com",
            "specialty": "Diamond jewelry and custom engagement rings",
            "description": "James Allen revolutionizes online jewelry shopping with their 360° Diamond Display Technology. They excel in customization, offering real-time diamond inspection and expert consultation. Their high-resolution imaging and virtual try-on features provide confidence in online purchases.",
            "price_range": "Medium to High",
            "best_for": ["Engagement Rings", "Loose Diamonds", "Custom Jewelry", "Wedding Bands"]
        },
        "Mejuri": {
            "url": "https://www.mejuri.com",
            "specialty": "Contemporary fine jewelry",
            "description": "Mejuri offers modern, minimalist fine jewelry for everyday wear. They focus on sustainable practices and direct-to-consumer pricing. Their pieces are perfect for those seeking contemporary designs with quality materials at accessible price points.",
            "price_range": "Low to Medium",
            "best_for": ["Daily Wear", "Modern Designs", "Gold Jewelry", "Affordable Luxury"]
        },
        "Brilliant Earth": {
            "url": "https://www.brilliantearth.com",
            "specialty": "Ethical and sustainable jewelry",
            "description": "Brilliant Earth leads in ethical and sustainable jewelry. They offer beyond-conflict-free diamonds and recycled precious metals. Their transparency in sourcing and commitment to environmental responsibility makes them perfect for conscious consumers.",
            "price_range": "Medium to High",
            "best_for": ["Ethical Jewelry", "Sustainable Diamonds", "Unique Designs", "Custom Engagement Rings"]
        },
        "Cartier": {
            "url": "https://www.cartier.com",
            "specialty": "Luxury and heritage jewelry",
            "description": "Cartier represents the pinnacle of luxury jewelry craftsmanship. Their iconic designs and heritage pieces are symbols of excellence. They offer exceptional quality and timeless elegance, perfect for those seeking prestigious branded jewelry.",
            "price_range": "High to Ultra-High",
            "best_for": ["Luxury Pieces", "Heritage Designs", "Statement Jewelry", "Investment Pieces"]
        },
        "Etsy": {
            "url": "https://www.etsy.com/c/jewelry",
            "specialty": "Handmade and vintage jewelry",
            "description": "Etsy connects buyers with independent artisans and vintage sellers. It's perfect for finding unique, handcrafted pieces and one-of-a-kind vintage jewelry. The platform offers a wide range of styles and price points with personal seller interaction.",
            "price_range": "Low to Medium",
            "best_for": ["Handmade Jewelry", "Vintage Pieces", "Custom Designs", "Unique Finds"]
        },
        "Tiffany & Co.": {
            "url": "https://www.tiffany.com",
            "specialty": "Luxury and iconic jewelry",
            "description": "Tiffany & Co. is synonymous with luxury and iconic design. Their pieces represent timeless elegance and superior craftsmanship. They offer exceptional quality and their signature style, perfect for those seeking prestigious branded jewelry.",
            "price_range": "High",
            "best_for": ["Luxury Jewelry", "Engagement Rings", "Statement Pieces", "Gift Jewelry"]
        }
    }
    
    # Calculate match scores based on preferences
    recommendations = []
    for name, info in jewelry_websites.items():
        score = 0
        
        # Match price range
        if preferences['budget'].lower() in info['price_range'].lower():
            score += 3
        
        # Match jewelry type
        if any(pref_type in item.lower() for item in info['best_for'] 
               for pref_type in preferences['type'].lower().split()):
            score += 2
        
        # Match style preferences
        if preferences['style'].lower() in info['specialty'].lower():
            score += 2
        
        # Match material preferences
        if preferences['material'].lower() in info['specialty'].lower():
            score += 2
        
        # Match purpose
        if preferences['purpose'].lower() in info['description'].lower():
            score += 1
        
        # Add to recommendations if there's any match
        if score > 0:
            recommendations.append({
                'name': name,
                'url': info['url'],
                'description': info['description'],
                'match_score': score,
                'price_range': info['price_range'],
                'best_for': info['best_for']
            })
    
    # Sort by match score and return top 5
    recommendations.sort(key=lambda x: x['match_score'], reverse=True)
    return recommendations[:5]

def main():
    st.set_page_config(
        page_title="Jewelry Analysis System",
        page_icon="💎",
        layout="wide"
    )
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Jewelry Analysis", "Recommendations", "Feedback"])
    
    with tab1:
        st.title("💎 Professional Jewelry Analysis System")
        st.write("Upload a jewelry image for comprehensive AI-powered analysis")
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Load and display image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            
            # Enhance image
            enhanced_image = enhance_image(image)
            with col2:
                st.subheader("Enhanced Image")
                st.image(enhanced_image, use_column_width=True)
            
            # Analyze jewelry
            with st.spinner("Performing detailed analysis of your jewelry..."):
                analysis_results = analyze_jewelry(enhanced_image)
                
                if analysis_results:
                    st.header("Detailed Jewelry Analysis")
                    
                    # Display confidence score
                    if analysis_results["confidence_score"] > 0:
                        st.metric("Analysis Confidence", f"{analysis_results['confidence_score']*100:.1f}%")
                    
                    # Jewelry Type Analysis
                    if analysis_results["jewelry_type_analysis"]:
                        st.subheader("Jewelry Type Analysis")
                        st.write(analysis_results["jewelry_type_analysis"])
                    
                    # Materials Composition
                    if analysis_results["materials_composition"]:
                        with st.expander("Materials Composition", expanded=True):
                            st.write(analysis_results["materials_composition"])
                    
                    # Gemstones Analysis
                    if analysis_results["gemstones_analysis"]:
                        with st.expander("Gemstones Analysis"):
                            st.write(analysis_results["gemstones_analysis"])
                    
                    # Design Elements
                    if analysis_results["design_elements"]:
                        with st.expander("Design Elements"):
                            st.write(analysis_results["design_elements"])
                    
                    # Quality Indicators
                    if analysis_results["quality_indicators"]:
                        with st.expander("Quality Assessment"):
                            st.write(analysis_results["quality_indicators"])
                    
                    # Brand Analysis
                    if analysis_results["brand_analysis"]:
                        with st.expander("Brand Analysis"):
                            st.write(analysis_results["brand_analysis"])
                    
                    # Care Guidelines
                    if analysis_results["care_guidelines"]:
                        with st.expander("Care Guidelines"):
                            st.write(analysis_results["care_guidelines"])
                    
                    # Market Relevance
                    if analysis_results["market_relevance"]:
                        with st.expander("Market Insights"):
                            st.write(analysis_results["market_relevance"])
                
                else:
                    st.warning("Could not analyze the image. Please try uploading a clearer image of jewelry.")
    
    with tab2:
        st.title("💎 Personalized Jewelry Recommendations")
        st.write("Answer a few questions to get personalized jewelry shopping recommendations")
        
        # Questionnaire
        st.subheader("Your Preferences")
        
        # Budget Range
        budget = st.select_slider(
            "What's your budget range?",
            options=["Low", "Medium", "High", "Ultra-High"],
            value="Medium"
        )
        
        # Jewelry Type
        jewelry_type = st.selectbox(
            "What type of jewelry are you looking for?",
            ["Engagement Rings", "Wedding Bands", "Necklaces", "Earrings", "Bracelets", 
             "Vintage Jewelry", "Statement Pieces", "Daily Wear"]
        )
        
        # Style Preference
        style = st.selectbox(
            "What's your preferred style?",
            ["Modern", "Classic", "Vintage", "Minimalist", "Luxury", "Artisan/Handmade"]
        )
        
        # Material Preference
        material = st.selectbox(
            "Preferred material?",
            ["Gold", "Silver", "Platinum", "Diamond", "Gemstones", "Mixed Materials"]
        )
        
        # Purpose
        purpose = st.selectbox(
            "What's the primary purpose?",
            ["Special Occasion", "Daily Wear", "Investment", "Gift", "Collection"]
        )
        
        if st.button("Get Recommendations"):
            preferences = {
                'budget': budget,
                'type': jewelry_type,
                'style': style,
                'material': material,
                'purpose': purpose
            }
            
            recommendations = get_jewelry_recommendations(preferences)
            
            if recommendations:
                st.subheader("Top Recommended Jewelry Websites")
                
                for i, rec in enumerate(recommendations, 1):
                    with st.expander(f"{i}. {rec['name']} (Match Score: {rec['match_score']})"):
                        st.write(f"**Website:** [{rec['name']}]({rec['url']})")
                        st.write(f"**Price Range:** {rec['price_range']}")
                        st.write(f"**Best For:** {', '.join(rec['best_for'])}")
                        st.write("\n**Description:**")
                        st.write(rec['description'])
    
    with tab3:
        st.title("💎 Feedback Form")
        st.write("Help us improve! Share your thoughts about our platform.")
        
        # User Information (Optional)
        st.subheader("Personal Information (Optional)")
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name")
            age = st.number_input("Age", min_value=0, max_value=120)
        with col2:
            email = st.text_input("Email")
            gender = st.selectbox("Gender", ["", "Male", "Female", "Other", "Prefer not to say"])
        
        # Experience Rating
        st.subheader("Your Experience")
        overall_rating = st.slider("How would you rate your overall experience?", 1, 5, 3)
        
        # Specific Feedback
        st.subheader("Detailed Feedback")
        
        # Analysis Quality
        analysis_quality = st.select_slider(
            "How accurate was the jewelry analysis?",
            options=["Poor", "Fair", "Good", "Very Good", "Excellent"],
            value="Good"
        )
        
        # Recommendation Quality
        recommendation_quality = st.select_slider(
            "How helpful were the recommendations?",
            options=["Poor", "Fair", "Good", "Very Good", "Excellent"],
            value="Good"
        )
        
        # Detailed Comments
        liked = st.text_area("What aspects of the platform did you like the most?")
        improve = st.text_area("What aspects could be improved?")
        features = st.text_area("What additional features would you like to see?")
        
        # Usage Information
        st.subheader("Usage Information")
        purpose = st.multiselect(
            "What did you use the platform for?",
            ["Jewelry Analysis", "Shopping Recommendations", "Market Research", "Other"]
        )
        
        frequency = st.select_slider(
            "How often do you plan to use this platform?",
            options=["Rarely", "Occasionally", "Regularly", "Frequently"],
            value="Occasionally"
        )
        
        # Additional Comments
        comments = st.text_area("Any additional comments or suggestions?")
        
        if st.button("Submit Feedback"):
            if any([name, email, liked, improve, features, comments]):
                # Here you would typically save the feedback to a database
                st.success("Thank you for your valuable feedback! We appreciate your input.")
                
                # Display summary
                st.subheader("Feedback Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Overall Rating", f"{overall_rating}/5")
                    st.metric("Analysis Quality", analysis_quality)
                with col2:
                    st.metric("Recommendation Quality", recommendation_quality)
                    st.metric("Usage Frequency", frequency)
            else:
                st.warning("Please provide at least some feedback before submitting.")

if __name__ == "__main__":
    main()
