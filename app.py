import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import PyPDF2
import io
import base64
import time
import threading
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from reportlab.lib import colors

# Load environment variables from .env file
load_dotenv()

# DIRECT APPROACH - Hardcoding the OpenAI client (same as in the simple app that works)
# The key from the .env file is working in our test script, so use the exact same approach
try:
    with open(".env", "r") as f:
        env_content = f.read()
except FileNotFoundError:
    print("Local .env file not found, trying to load environment variables directly")
    env_content = os.environ.get("OPENAI_API_KEY", "")
    
# Parse the API key
API_KEY = None
if isinstance(env_content, str):
    if "OPENAI_API_KEY=" in env_content:
        # Parse from .env file format
        raw_key = env_content.split("OPENAI_API_KEY=")[1].strip()
        # Clean any quotes or newlines
        API_KEY = raw_key.strip().strip("'").strip('"').strip()
    else:
        # Directly from environment variable
        API_KEY = env_content

if API_KEY:
    # Mask the key for display
    if len(API_KEY) > 8:
        print(f"API key loaded: {API_KEY[:4]}...{API_KEY[-4:]}")
    print(f"API key length: {len(API_KEY)}")
    client = OpenAI(api_key=API_KEY)
    
    # Test the API key quickly - Skip test in development mode
    if API_KEY != "dummy_key_for_testing":
        try:
            test_response = client.chat.completions.create(
                model="gpt-4.5-preview",
                messages=[
                    {"role": "user", "content": "Say OK"}
                ],
                max_tokens=5
            )
            print(f"API key test successful: {test_response.choices[0].message.content}")
        except Exception as e:
            print(f"API key test failed: {str(e)}")
            API_KEY = None
            client = None
    else:
        print("Using dummy API key for testing - skipping API test")
else:
    print("No API key found")
    API_KEY = None
    client = None

print("\n" + "-"*50)
print("STARTUP ANALYSIS AND PLANNING APP")
print("-"*50)
print("App is running with API key from .env file")
print("-"*50 + "\n")

# Read the prompts from the specified file
def load_prompts(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    
    # Different parsing logic for different file formats
    if file_path == "startup_analysis_prompts.txt":
        return parse_analysis_prompts(content)
    elif file_path == "startup_plans_prompts.txt":
        return parse_plan_prompts(content)
    elif file_path == "ai_research_paper_prompts.txt":
        return parse_research_prompts(content)
    elif file_path == "neurips_review.txt":
        return parse_neurips_prompts(content)
    elif file_path == "iclr_review.txt":
        return parse_iclr_prompts(content)
    else:
        print(f"Unknown prompt file: {file_path}")
        return {}

# Parse the analysis prompts format
def parse_analysis_prompts(content):
    # Split the content into sections
    sections = {}
    current_section = None
    current_prompts = []
    
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('### '):
            if current_section:
                sections[current_section] = current_prompts
            current_section = line.replace('### ', '').strip()
            current_prompts = []
        elif line.startswith('**') and '**' in line and '.' in line:
            # This is a prompt title
            parts = line.split('.')
            prompt_number = parts[0].replace('**', '').strip()
            if len(parts) > 1:
                prompt_title = parts[1].replace('**', '').strip()
                
                # Find the full prompt content
                i += 1
                prompt_content = ""
                while i < len(lines) and not (lines[i].startswith('**') or lines[i].startswith('---') or lines[i].startswith('###')):
                    if lines[i].strip():
                        prompt_content += lines[i] + '\n'
                    i += 1
                i -= 1  # Adjust for next iteration
                
                current_prompts.append((prompt_number, prompt_title, prompt_content))
        i += 1
    
    # Add the last section
    if current_section:
        sections[current_section] = current_prompts
    
    return sections

# Parse the plan prompts format
def parse_plan_prompts(content):
    # Split the content into sections
    sections = {}
    current_section = None
    current_prompts = []
    
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        # Check for section headers which use ## pattern
        if line.startswith('## '):
            if current_section:
                sections[current_section] = current_prompts
            current_section = line.replace('## ', '').strip().replace('**', '')
            current_prompts = []
        # Check for prompt headers which use ### pattern
        elif line.startswith('### '):
            prompt_line = line.replace('### ', '').strip().replace('**', '')
            
            # Extract prompt number and title
            if "**" in prompt_line:
                prompt_line = prompt_line.replace('**', '')
            
            # Parse the prompt number and title
            if '. ' in prompt_line:
                parts = prompt_line.split('. ', 1)
                prompt_number = parts[0]
                prompt_title = parts[1] if len(parts) > 1 else prompt_line
            else:
                prompt_number = ""
                prompt_title = prompt_line
            
            # Find the prompt content (usually on the next line)
            i += 1
            prompt_content = ""
            # Plans prompts content is typically enclosed in > symbols
            while i < len(lines) and lines[i].strip() and not (lines[i].startswith('##') or lines[i].startswith('---')):
                content_line = lines[i].strip()
                if content_line.startswith('>'):
                    content_line = content_line[1:].strip()  # Remove the > and any space
                prompt_content += content_line + '\n'
                i += 1
            i -= 1  # Adjust for next iteration
            
            current_prompts.append((prompt_number, prompt_title, prompt_content))
        i += 1
    
    # Add the last section
    if current_section:
        sections[current_section] = current_prompts
    
    return sections

# Parse the research paper prompts format
def parse_research_prompts(content):
    # Split the content into sections
    sections = {}
    current_section = None
    current_prompts = []
    
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        # Check for section headers which use ## pattern
        if line.startswith('## '):
            if current_section:
                sections[current_section] = current_prompts
            current_section = line.replace('## ', '').strip()
            current_prompts = []
        # Check for subsection headers which use ### pattern 
        elif line.startswith('### '):
            prompt_title = line.replace('### ', '').strip()
            prompt_number = ""  # No specific numbering in research paper format
            
            # Find the prompt content (bullet points that follow)
            i += 1
            prompt_content = ""
            while i < len(lines) and not (lines[i].startswith('##') or lines[i].startswith('# ')):
                if lines[i].strip():
                    # Include all bullet points and text under this subsection
                    prompt_content += lines[i] + '\n'
                i += 1
            i -= 1  # Adjust for next iteration
            
            # Add to the current section
            if prompt_content.strip():
                current_prompts.append((prompt_number, prompt_title, prompt_content))
        i += 1
    
    # Add the last section
    if current_section:
        sections[current_section] = current_prompts
    
    return sections

# Parse the NeurIPS review prompts format
def parse_neurips_prompts(content):
    # Split the content into sections
    sections = {"NeurIPS Review": []}
    current_prompts = []
    
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Look for numbered prompts (e.g. "1. Briefly summarize...")
        if line and line[0].isdigit() and ". " in line:
            parts = line.split(". ", 1)
            prompt_number = parts[0]
            prompt_title = parts[1]
            
            # Get the content (everything until the next numbered prompt)
            i += 1
            prompt_content = ""
            while i < len(lines) and not (i < len(lines) and lines[i].strip() and lines[i].strip()[0].isdigit() and ". " in lines[i].strip()):
                if lines[i].strip():
                    prompt_content += lines[i] + '\n'
                i += 1
            i -= 1  # Adjust for next iteration
            
            # Create a properly formatted prompt that will work with the paper text
            formatted_prompt = f"""Based on the following research paper, {prompt_title}

Paper text:
<idea>

Please provide a detailed response addressing this aspect of the review."""
            
            # Add to the prompts
            current_prompts.append((prompt_number, prompt_title, formatted_prompt))
        i += 1
    
    # Add all prompts to the NeurIPS Review section
    sections["NeurIPS Review"] = current_prompts
    
    return sections

# Parse the ICLR review prompts format
def parse_iclr_prompts(content):
    # Split the content into sections
    sections = {"ICLR Review": []}
    current_prompts = []
    
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Look for numbered prompts (e.g. "1. Summarize...")
        if line and line[0].isdigit() and ". " in line:
            parts = line.split(". ", 1)
            prompt_number = parts[0]
            prompt_title = parts[1]
            
            # Get the content (everything until the next numbered prompt)
            i += 1
            prompt_content = ""
            while i < len(lines) and not (i < len(lines) and lines[i].strip() and lines[i].strip()[0].isdigit() and ". " in lines[i].strip()):
                if lines[i].strip():
                    prompt_content += lines[i] + '\n'
                i += 1
            i -= 1  # Adjust for next iteration
            
            # Create a properly formatted prompt that will work with the paper text
            formatted_prompt = f"""Based on the following research paper, {prompt_title}

Paper text:
<idea>

Please provide a detailed response addressing this aspect of the review."""
            
            # Add to the prompts
            current_prompts.append((prompt_number, prompt_title, formatted_prompt))
        i += 1
    
    # Add all prompts to the ICLR Review section
    sections["ICLR Review"] = current_prompts
    
    return sections

# Function to generate a PDF from all the responses
def generate_pdf(results, all_prompts):
    # Create a BytesIO buffer to receive the PDF data
    buffer = io.BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Add custom styles - use try/except to avoid errors if styles already exist
    try:
        styles.add(ParagraphStyle(name='CustomHeading1', 
                                 fontName='Helvetica-Bold',
                                 fontSize=16,
                                 spaceAfter=12,
                                 textColor=colors.blue))
    except KeyError:
        # If style already exists, just modify it
        styles['CustomHeading1'] = ParagraphStyle(name='CustomHeading1',
                                 fontName='Helvetica-Bold',
                                 fontSize=16,
                                 spaceAfter=12,
                                 textColor=colors.blue)
    
    try:
        styles.add(ParagraphStyle(name='CustomHeading2', 
                                 fontName='Helvetica-Bold',
                                 fontSize=14,
                                 spaceAfter=10,
                                 textColor=colors.darkblue))
    except KeyError:
        styles['CustomHeading2'] = ParagraphStyle(name='CustomHeading2',
                                 fontName='Helvetica-Bold',
                                 fontSize=14,
                                 spaceAfter=10,
                                 textColor=colors.darkblue)
    
    try:
        styles.add(ParagraphStyle(name='CustomNormal',
                                 fontName='Helvetica',
                                 fontSize=10,
                                 spaceAfter=10))
    except KeyError:
        styles['CustomNormal'] = ParagraphStyle(name='CustomNormal',
                                 fontName='Helvetica',
                                 fontSize=10,
                                 spaceAfter=10)
    
    # Create the content
    content = []
    
    # Add title with date
    title_style = styles["Title"]  # Use built-in Title style
    title_style.alignment = TA_CENTER
    title = f"Analysis Report - {datetime.now().strftime('%Y-%m-%d')}"
    content.append(Paragraph(title, title_style))
    content.append(Spacer(1, 20))
    
    # Process all responses
    current_section = None
    
    for idx, prompt_info in enumerate(all_prompts):
        # Only include prompts that have responses
        if idx not in results:
            continue
            
        section_name, num, title, _ = prompt_info
        response_text = results[idx]
        
        # Add section header if we're in a new section
        if section_name != current_section:
            content.append(Spacer(1, 10))
            content.append(Paragraph(section_name, styles["CustomHeading1"]))
            content.append(Spacer(1, 10))
            current_section = section_name
        
        # Add the prompt number and title
        prompt_title = f"{num}. {title}"
        content.append(Paragraph(prompt_title, styles["CustomHeading2"]))
        
        # Process and add the response
        # First, remove any existing HTML tags that might cause problems
        response_text = response_text.replace("<para>", "").replace("</para>", "")
        response_text = response_text.replace("<b>", "").replace("</b>", "")
        response_text = response_text.replace("<br>", "\n").replace("<br/>", "\n")
        
        # Now apply our own formatting
        # Replace markdown elements with reportlab styling
        response_text = response_text.replace("###", "")
        response_text = response_text.replace("##", "")
        
        # Special handling for markdown tables
        if "|" in response_text:
            # This is a markdown table - we need special handling
            try:
                # First, treat the table separately from rest of text
                sections = []
                current_section = ""
                in_table = False
                
                lines = response_text.split("\n")
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    
                    # Check if this line could be the start of a table
                    if "|" in line:
                        # Look ahead to see if next line contains table header separator
                        is_table_start = False
                        for j in range(i+1, min(i+3, len(lines))):
                            next_line = lines[j].strip()
                            # Check for various Markdown table header separator formats
                            if ("|" in next_line and 
                                ("-|-" in next_line or 
                                 "---|" in next_line or 
                                 "|---" in next_line or
                                 ":--" in next_line or
                                 "--:" in next_line)):
                                is_table_start = True
                                break
                        
                        if is_table_start:
                            # End previous text section
                            if current_section:
                                sections.append(("text", current_section))
                                current_section = ""
                            
                            # Start collecting table content
                            in_table = True
                            table_content = line + "\n"
                            i += 1
                            continue
                    
                    # Process based on whether we're in a table or not
                    if in_table:
                        if "|" in line:
                            # Still part of the table
                            table_content += line + "\n"
                        else:
                            # Empty line might still be part of table formatting
                            if not line.strip():
                                i += 1
                                # Check if next line has pipes (still table)
                                if i < len(lines) and "|" in lines[i]:
                                    table_content += "\n" + lines[i] + "\n"
                                    continue
                            
                            # End of table reached
                            sections.append(("table", table_content))
                            in_table = False
                            current_section = line + "\n" if line else ""
                    else:
                        # Regular text
                        current_section += line + "\n"
                    
                    i += 1
                
                # Add the last section
                if in_table:
                    sections.append(("table", table_content))
                elif current_section:
                    sections.append(("text", current_section))
                
                # Process each section
                for section_type, section_content in sections:
                    if section_type == "text":
                        # Process regular text with improved formatting for bullet points
                        lines = section_content.split("\n")
                        processed_paras = []
                        current_para = []
                        in_bullet_list = False
                        
                        for line in lines:
                            line_stripped = line.strip()
                            
                            # Check if line is empty - potential paragraph break
                            if not line_stripped:
                                # End current paragraph if we have content
                                if current_para:
                                    processed_paras.append("\n".join(current_para))
                                    current_para = []
                                    in_bullet_list = False
                                continue
                            
                            # Check for bullet points
                            if line_stripped.startswith("- ") or line_stripped.startswith("* "):
                                # If we're starting a new bullet list, end previous paragraph
                                if not in_bullet_list and current_para:
                                    processed_paras.append("\n".join(current_para))
                                    current_para = []
                                
                                # Format bullet point
                                if line_stripped.startswith("- "):
                                    bullet_text = "• " + line_stripped[2:]
                                else:  # starts with *
                                    bullet_text = "• " + line_stripped[2:]
                                
                                # Add bullet point
                                current_para.append(bullet_text)
                                in_bullet_list = True
                            else:
                                # Regular text line
                                # If we were in a bullet list, end it
                                if in_bullet_list:
                                    processed_paras.append("\n".join(current_para))
                                    current_para = []
                                    in_bullet_list = False
                                
                                # Add regular text line
                                current_para.append(line_stripped)
                        
                        # Add final paragraph if any
                        if current_para:
                            processed_paras.append("\n".join(current_para))
                        
                        # Create paragraphs with proper styling
                        for paragraph in processed_paras:
                            if not paragraph.strip():
                                continue
                                
                            try:
                                # Check if this is a bullet list
                                if "•" in paragraph:
                                    # Special style for bullet points with extra spacing
                                    bullet_style = ParagraphStyle(
                                        "BulletStyle", 
                                        parent=styles["CustomNormal"],
                                        leftIndent=10,
                                        leading=14  # Line spacing for bullets
                                    )
                                    content.append(Paragraph(paragraph, bullet_style))
                                else:
                                    # Regular paragraph
                                    content.append(Paragraph(paragraph, styles["CustomNormal"]))
                                
                                # Add space between paragraphs
                                content.append(Spacer(1, 8))
                            except Exception as e:
                                print(f"Error adding paragraph: {str(e)}")
                                content.append(Paragraph("Error formatting content", styles["CustomNormal"]))
                    
                    elif section_type == "table":
                        # Import only Table and TableStyle, don't reimport colors
                        from reportlab.platypus import Table, TableStyle
                        
                        # Clean and preprocess the markdown table text
                        table_text = section_content.strip()
                        
                        # Clean up common markdown table formatting issues
                        # 1. Fix situations where content is split across lines improperly
                        lines = table_text.split('\n')
                        cleaned_lines = []
                        current_line = ""
                        
                        for line in lines:
                            stripped = line.strip()
                            if not stripped:
                                continue
                                
                            # If line has pipes and doesn't look like a header separator
                            if "|" in stripped:
                                # If it's a header separator line, add it as is
                                if any(sep in stripped for sep in ["---", ":-:", "-|-", ":--", "--:"]):
                                    # Add previous constructed line if any
                                    if current_line:
                                        cleaned_lines.append(current_line)
                                        current_line = ""
                                    # Add separator line
                                    cleaned_lines.append(stripped)
                                # Otherwise it's a content line
                                else:
                                    # Check if it's a complete row with balanced pipes
                                    if stripped.startswith("|") and stripped.endswith("|"):
                                        # Complete line with matching first and last pipes
                                        if current_line:
                                            cleaned_lines.append(current_line)
                                            current_line = ""
                                        cleaned_lines.append(stripped)
                                    else:
                                        # Incomplete line or line continuation - append to current line
                                        if not current_line:
                                            current_line = stripped
                                        else:
                                            current_line += " " + stripped
                        
                        # Add any remaining line
                        if current_line:
                            cleaned_lines.append(current_line)
                        
                        # Ensure proper table structure with pipes at start/end
                        for i in range(len(cleaned_lines)):
                            line = cleaned_lines[i]
                            # Add missing pipes at beginning/end if needed
                            if not line.startswith("|"):
                                cleaned_lines[i] = "|" + line
                            if not line.endswith("|"):
                                cleaned_lines[i] = cleaned_lines[i] + "|"
                                                
                        # Now extract the actual table data
                        data_rows = []
                        header_row = None
                        
                        for i, line in enumerate(cleaned_lines):
                            # Skip empty lines
                            if not line.strip():
                                continue
                                
                            # Skip separator rows (the ones with ---)
                            if any(sep in line for sep in ["---", ":-:", "-|-", ":--", "--:"]):
                                continue
                                
                            # Process table row - split by pipes and clean
                            cells = [cell.strip() for cell in line.split("|")]
                            # Remove empty cells from start/end (which come from the outer pipes)
                            cells = [cell for cell in cells if cell != ""]
                            
                            if cells:
                                if header_row is None:
                                    header_row = cells
                                else:
                                    data_rows.append(cells)
                        
                        # Create table data with header and rows
                        if header_row and len(header_row) > 0:
                            # Create table data by combining header and data rows
                            table_data = [header_row]
                            if data_rows:
                                table_data.extend(data_rows)
                            
                            # Create the table
                            try:
                                # Make sure all rows have the same number of columns
                                max_cols = max(len(row) for row in table_data)
                                for row in table_data:
                                    while len(row) < max_cols:
                                        row.append("")
                                
                                # Calculate column widths based on content
                                col_widths = [None] * max_cols
                                
                                # Get available width (letter page width minus margins)
                                available_width = 500  # Approximate available width in points
                                
                                # Create and style the table with adjusted width
                                table = Table(table_data, colWidths=col_widths, repeatRows=1)
                                table.setStyle(TableStyle([
                                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                    ('FONTSIZE', (0, 0), (-1, -1), 8),  # Smaller font for tables
                                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                    ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
                                    ('LEFTPADDING', (0, 0), (-1, -1), 2),  # Reduce cell padding
                                    ('RIGHTPADDING', (0, 0), (-1, -1), 2),  # Reduce cell padding
                                    ('TOPPADDING', (0, 0), (-1, -1), 3),    # Reduce cell padding
                                    ('BOTTOMPADDING', (0, 0), (-1, -1), 3), # Reduce cell padding
                                    ('WORDWRAP', (0, 0), (-1, -1), True),   # Enable word wrapping
                                ]))
                                
                                # Add to content
                                content.append(table)
                                content.append(Spacer(1, 12))
                            except Exception as e:
                                print(f"Error creating table: {str(e)}")
                                # Fallback to plain text version
                                content.append(Paragraph("Table formatting error. Displaying as text:", styles["CustomNormal"]))
                                for row in table_data:
                                    content.append(Paragraph(" | ".join(row), styles["CustomNormal"]))
                        else:
                            # Fallback if no proper table structure found
                            content.append(Paragraph("Table could not be properly formatted:", styles["CustomNormal"]))
                            for line in table_lines:
                                if line.strip() and not ("---" in line and "|" in line):
                                    content.append(Paragraph(line, styles["CustomNormal"]))
                        
                        content.append(Spacer(1, 12))
            
            except Exception as e:
                print(f"Error processing table: {str(e)}")
                # Fallback: just replace pipes and treat as plain text
                response_text = response_text.replace("|", " | ")
                content.append(Paragraph(response_text, styles["CustomNormal"]))
        
        else:
            # No tables, process as regular text
            # Handle bullet points and paragraphs properly
            lines = response_text.split("\n")
            processed_paras = []
            current_para = []
            in_bullet_list = False
            
            for line in lines:
                line_stripped = line.strip()
                
                # Check if line is empty - potential paragraph break
                if not line_stripped:
                    # End current paragraph if we have content
                    if current_para:
                        processed_paras.append("\n".join(current_para))
                        current_para = []
                        in_bullet_list = False
                    continue
                
                # Check for bullet points
                if line_stripped.startswith("- ") or line_stripped.startswith("* "):
                    # If we're starting a new bullet list, end previous paragraph
                    if not in_bullet_list and current_para:
                        processed_paras.append("\n".join(current_para))
                        current_para = []
                    
                    # Format bullet point
                    if line_stripped.startswith("- "):
                        bullet_text = "• " + line_stripped[2:]
                    else:  # starts with *
                        bullet_text = "• " + line_stripped[2:]
                    
                    # Add bullet point
                    current_para.append(bullet_text)
                    in_bullet_list = True
                else:
                    # Regular text line
                    # If we were in a bullet list, end it
                    if in_bullet_list:
                        processed_paras.append("\n".join(current_para))
                        current_para = []
                        in_bullet_list = False
                    
                    # Add regular text line
                    current_para.append(line_stripped)
            
            # Add final paragraph if any
            if current_para:
                processed_paras.append("\n".join(current_para))
            
            # Create paragraphs with proper styling
            for paragraph in processed_paras:
                if not paragraph.strip():
                    continue
                    
                try:
                    # Check if this is a bullet list
                    if "•" in paragraph:
                        # Special style for bullet points with extra spacing
                        bullet_style = ParagraphStyle(
                            "BulletStyle", 
                            parent=styles["CustomNormal"],
                            leftIndent=10,
                            leading=14  # Line spacing for bullets
                        )
                        content.append(Paragraph(paragraph, bullet_style))
                    else:
                        # Regular paragraph
                        content.append(Paragraph(paragraph, styles["CustomNormal"]))
                    
                    # Add space between paragraphs
                    content.append(Spacer(1, 8))
                except Exception as e:
                    print(f"Error adding paragraph: {str(e)}")
                    # Fallback: add as plain text without any formatting
                    content.append(Paragraph("Error formatting content", styles["CustomNormal"]))
        content.append(Spacer(1, 15))
    
    # Build the PDF
    doc.build(content)
    
    # Get the PDF data
    pdf_data = buffer.getvalue()
    buffer.close()
    
    return pdf_data

# Function to create a download link for the generated PDF
def get_pdf_download_link(pdf_data, filename="report.pdf"):
    b64 = base64.b64encode(pdf_data).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Download PDF Report</a>'
    return href

# Function to call OpenAI API with streaming
def call_openai_api(prompt, idea, current_idx, all_prompts, results, placeholder=None):
    global API_KEY, client
    
    # Format the prompt first
    formatted_prompt = prompt.replace('<idea>', idea)
    
    # Error message to display if there's no API key
    api_key_error_msg = """
    Error: No valid OpenAI API key found. Please:
    
    1. Get a valid API key from https://platform.openai.com/api-keys
    2. Add it to your .env file as OPENAI_API_KEY=your_key
    3. Restart the app
    """
    
    # Check if we have a valid API key
    if not API_KEY:
        print("DEBUG: No valid API key available")
        return api_key_error_msg
    
    try:
        # Create a new client for each API call to ensure we're using the correct key
        direct_client = OpenAI(api_key=API_KEY)
        
        # Build message history with all previous prompts and responses
        messages = [
            {"role": "system", "content": """You are a startup analysis expert. Provide detailed, data-driven responses. Build upon previous analyses in your responses.
            
Format your output using Markdown:
1. Use tables to organize information when presenting comparative data, metrics, or segments
2. IMPORTANT: When formatting tables:
   - Ensure each row has the same number of cells/columns (always match the header)
   - Do not use bullet points inside table cells (use simple text instead)
   - Keep table cell content concise (use short phrases or single words when possible)
   - Use clear column headers in the first row
   - Each row must be on a single line (no line breaks inside a row)
   - Format tables properly with header separator rows using this exact format: | --- | --- | --- |
   - Make sure columns are properly aligned with | at beginning and end of each row
   - For wide tables with many columns, consider breaking into multiple smaller tables
   - Use proper markdown table syntax with pipes and dashes
   - Keep table width to max 5-6 columns for readability
3. Use bullet points for lists and key points (outside of tables)
   - Place an empty line before and after bullet point lists
   - Start each bullet with a single hyphen (-)
4. Use headers (### or ####) to organize sections
5. Clearly label all sections and tables"""}
        ]
        
        # Add all previous prompts and responses as context
        for i in range(current_idx):
            if i in results:
                section_name, num, title, prev_prompt = all_prompts[i]
                prev_formatted = prev_prompt.replace('<idea>', idea)
                
                # Add the previous prompt
                messages.append({"role": "user", "content": f"Previous analysis request {i+1}: {prev_formatted}"})
                
                # Add the previous response
                messages.append({"role": "assistant", "content": f"Previous analysis result {i+1}: {results[i]}"})
        
        # Add the current prompt
        messages.append({"role": "user", "content": f"Based on all previous analyses, please provide the next analysis: {formatted_prompt}"})
        
        print(f"Making streaming API call with key: {API_KEY[:4]}...{API_KEY[-4:]}")
        
        # Use the placeholder passed in, or create a new one if none
        try:
            # Use provided placeholder or create a new one
            response_placeholder = placeholder if placeholder is not None else st.empty()
            full_response = ""
            
            # Use streaming for real-time updates with longer timeout
            stream = direct_client.chat.completions.create(
                model="gpt-4.5-preview",
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                stream=True,  # Enable streaming
                timeout=600  # Set 10-minute timeout for API call
            )
            
            # Process the stream
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    # Just replace <br> tags with spaces
                    processed_response = full_response.replace("<br>", " ")
                    
                    # Update the display with each processed chunk
                    response_placeholder.markdown(processed_response)
        except Exception as e:
            print(f"Streaming error: {str(e)}")
            full_response = "Error during streaming. Please try again."
        
        return full_response
    except Exception as e:
        error_str = str(e)
        print(f"DEBUG: API error: {error_str}")
        
        if "insufficient_quota" in error_str:
            return """
            Error: Your OpenAI account has insufficient quota or credits.
            
            Please visit https://platform.openai.com/account/billing to add credits to your account.
            Once you've added credits, restart the app to continue.
            """
        elif "invalid_api_key" in error_str:
            return api_key_error_msg
        else:
            return f"Error: {error_str}"

# Keep the Streamlit connection alive through a heartbeat
def keep_connection_alive():
    # This function runs in a separate thread and periodically 
    # updates a hidden value to keep the connection alive
    while True:
        # Update heartbeat counter every 15 seconds
        if 'heartbeat' in st.session_state:
            st.session_state.heartbeat += 1
        time.sleep(15)

# Main Streamlit app
def main():
    global API_KEY, client
    
    # Initialize heartbeat counter
    if 'heartbeat' not in st.session_state:
        st.session_state.heartbeat = 0
        # Start the heartbeat thread
        threading.Thread(target=keep_connection_alive, daemon=True).start()
    
    # Increase session timeout to prevent the app from closing too soon
    # These settings can also be set in .streamlit/config.toml
    st.set_page_config(
        page_title="Startup Analysis & Planning",
        page_icon="🚀",
        layout="wide"
    )
    
    # Set longer server timeout - helps prevent disconnections
    import streamlit.runtime.scriptrunner.script_runner as script_runner
    # Try to increase the timeout to 5 minutes (300 seconds)
    try:
        script_runner.get_script_run_ctx().session_state["_script_run_ctx"]._timeout = 300
    except Exception as e:
        # Just continue if we can't modify the timeout
        print(f"Could not set custom timeout: {e}")
    
    # Settings moved to .streamlit/config.toml file for persistence
    
    # Simple styling - no custom CSS
    
    st.title("Startup Analysis and Planning")
    
    # Don't show API key field in GUI - use .env file or environment variables instead
    
    # Initialize session state if not exists
    if 'current_prompt_index' not in st.session_state:
        st.session_state.current_prompt_index = 0
        
    if 'results' not in st.session_state:
        st.session_state.results = {}
        
    if 'idea' not in st.session_state:
        st.session_state.idea = ""
        
    if 'mode' not in st.session_state:
        st.session_state.mode = None  # 'analyze' or 'plan'
        
    # Track which results have been seen (for debugging)
    if 'seen_results' not in st.session_state:
        st.session_state.seen_results = set()
    
    # Text input
    idea = st.text_area("Idea input", 
                     placeholder="Enter your idea here", 
                     height=100,
                     key="main_idea_input",
                     label_visibility="hidden")
    
    # Make sure we still have a valid idea value, even if the text area is empty
    # This handles the case where the user has entered text, then clicked a button
    if not idea and 'idea' in st.session_state:
        idea = st.session_state.idea
    
    # Buttons in a row
    button_col1, button_col2, button_space = st.columns([1, 1, 4])
    with button_col1:
        analyze_button = st.button("Analyze", use_container_width=True)

    with button_col2:
        plan_button = st.button("Plan", use_container_width=True)
    
    # Get input text (either from text area or PDF)
    # If text input is empty but we have text from PDF, use the PDF text
    input_text = idea if idea else st.session_state.get('idea', '')
    
    # Process submission
    if analyze_button:
        if not input_text or input_text.strip() == "":
            st.error("Please enter text or upload a PDF file first.")
        else:
            # Save idea to session state
            st.session_state.idea = input_text
            st.session_state.mode = "analyze"
            
            # Clear ALL previous results on new submission
            st.session_state.results = {}
            st.session_state.seen_results = set()
            
            # Reset to first prompt
            st.session_state.current_prompt_index = 0
            
            # Rerun the app to reset everything
            st.rerun()
            
    elif plan_button:
        if not input_text or input_text.strip() == "":
            st.error("Please enter text or upload a PDF file first.")
        else:
            # Save idea to session state
            st.session_state.idea = input_text
            st.session_state.mode = "plan"
            
            # Clear ALL previous results on new submission
            st.session_state.results = {}
            st.session_state.seen_results = set()
            
            # Reset to first prompt
            st.session_state.current_prompt_index = 0
            
            # Rerun the app to reset everything
            st.rerun()
            

    
    # Only continue if user has selected a mode
    if not st.session_state.mode:
        return
        
    # Load appropriate prompts based on mode
    if st.session_state.mode == "analyze":
        prompts_sections = load_prompts("startup_analysis_prompts.txt")
    elif st.session_state.mode == "plan":
        prompts_sections = load_prompts("startup_plans_prompts.txt")
    elif st.session_state.mode == "neurips":
        prompts_sections = load_prompts("neurips_review.txt")
    elif st.session_state.mode == "iclr":
        prompts_sections = load_prompts("iclr_review.txt")
    else:  # research mode
        prompts_sections = load_prompts("ai_research_paper_prompts.txt")
    
    # Flatten prompts into a sequential list
    all_prompts = []
    for section_name, prompts in prompts_sections.items():
        for num, title, content in prompts:
            all_prompts.append((section_name, num, title, content))
    
    # Get current prompt
    if all_prompts:
        current_idx = st.session_state.current_prompt_index
        
        # No progress indicator shown
        
        # No duplicate input field or submit button here
        # Just show which stage we're currently on with themed icons
        _, _, _, content = all_prompts[current_idx]
        section_name, num, title = all_prompts[current_idx][0:3]
        
        # Use only swimming/biking/running icons based on progress (3 icons total)
        total_prompts = len(all_prompts)
        if current_idx < total_prompts / 3:
            icon = "🏊"  # Swimming (early stage) - using simpler swimming icon
        elif current_idx < 2 * total_prompts / 3:
            icon = "🚴"  # Biking (middle stage) - using simpler biking icon
        else:
            icon = "🏃"  # Running (final stage) - using simpler running icon
            
        st.caption(f"{icon} {section_name}: {num}. {title}")
        
        # Use the idea already stored in session state
        idea = st.session_state.idea
        
        # Create a single placeholder for response display
        result_placeholder = st.empty()
            
        # Handle result display (and auto-generation if needed)
        if current_idx in st.session_state.results:
            # We already have a result for this index - display it
            # This avoids re-streaming responses we've already generated
            st.session_state.seen_results.add(current_idx)
            # Get the content and process it
            markdown_content = st.session_state.results[current_idx]
            
            # Just replace <br> tags with spaces
            markdown_content = markdown_content.replace("<br>", " ")
            
            # Render as plain markdown without any HTML
            result_placeholder.markdown(markdown_content)
        elif st.session_state.idea:
            # No result exists yet, but we have an idea - generate it
            # Get prompt details
            _, _, _, content = all_prompts[current_idx]
            
            try:
                # Auto-generate response using streaming with our placeholder
                # This will display the response as it's generated
                result = call_openai_api(
                    prompt=content,
                    idea=st.session_state.idea,
                    current_idx=current_idx,
                    all_prompts=all_prompts,
                    results=st.session_state.results,
                    placeholder=result_placeholder
                )
                
                # Just store the result for future navigation, don't display again
                # The streaming has already displayed it in the placeholder
                st.session_state.results[current_idx] = result
            except Exception as e:
                error_message = f"Error generating response: {str(e)}"
                print(error_message)
                result_placeholder.error(error_message)
                st.session_state.results[current_idx] = error_message
            
        # Debug info - not visible to user but helpful for developers
        # Show which responses we have in memory
        print(f"Current index: {current_idx}")
        print(f"Responses in memory: {list(st.session_state.results.keys())}")
        print(f"Seen responses: {list(st.session_state.seen_results)}")
            
        # Always show navigation buttons, even before result is displayed
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if current_idx > 0:
                if st.button("Back"):
                    # Simply go back to previous result (which should already be generated)
                    st.session_state.current_prompt_index -= 1
                    st.rerun()
        
        # Only show the Generate PDF button if we have some results
        with col3:
            if st.session_state.results:
                if st.button("Generate PDF Report"):
                    # Generate PDF with all responses
                    pdf_data = generate_pdf(st.session_state.results, all_prompts)
                    
                    # Create a filename based on mode
                    mode = st.session_state.mode.capitalize()
                    filename = f"{mode}_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    
                    # Create download link
                    st.markdown(get_pdf_download_link(pdf_data, filename), unsafe_allow_html=True)
        
        # Show Next button if not on the last prompt
        with col2:
            if current_idx < len(all_prompts) - 1:
                if st.button("Next"):
                    # Move to next prompt
                    next_idx = current_idx + 1
                    st.session_state.current_prompt_index = next_idx
                    
                    # If we have an idea and the next response hasn't been generated yet
                    # We don't need to generate it here - the main display logic will 
                    # handle showing an empty container, and the API call will be made when shown
                    
                    # Rerun to show the updated UI
                    st.rerun()
    
    # Footer
    st.markdown("---")
    footer1, footer2 = st.columns([3, 1])
    with footer1:
        st.markdown("Made with ❤️ by Claude Code and running OpenAI GPT 4.5")
    with footer2:
        # Show heartbeat in small text - helps monitor connection status
        st.caption(f"Connection heartbeat: {st.session_state.heartbeat}")

if __name__ == "__main__":
    main()
