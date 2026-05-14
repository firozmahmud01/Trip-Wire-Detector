import requests
from requests.auth import HTTPDigestAuth
import xml.etree.ElementTree as ET

IP_ADDRESS = '192.168.1.100' 
PORT = 80            
USER = 'admin'
PASS = '123456789!' 

PROFILE_TOKEN = "Profile_1" 


PTZ_URL = f"http://{IP_ADDRESS}:{PORT}/onvif/ptz_service"

def send_ptz_command(xml_body):
    """Sends the SOAP envelope to the camera."""

    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
    }
    try:
        response = requests.post(
            PTZ_URL, 
            data=xml_body, 
            headers=headers, 
            auth=HTTPDigestAuth(USER, PASS),
            timeout=1
        )
        return response.status_code
    except Exception as e:
        print(f"Connection Error: {e}")
        return None

def movecamera(direction, velocity=0.5):
    """Moves the camera: up, down, left, right."""
    # Mapping directions to X (Pan) and Y (Tilt) vectors
    vectors = {
        "up":    {'x': 0, 'y': velocity},
        "down":  {'x': 0, 'y': -velocity},
        "left":  {'x': -velocity, 'y': 0},
        "right": {'x': velocity, 'y': 0}
    }
    
    v = vectors.get(direction, {'x': 0, 'y': 0})
    
    soap_body = f"""<?xml version="1.0" encoding="UTF-8"?>
    <s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl" xmlns:tt="http://www.onvif.org/ver10/schema">
      <s:Body>
        <tptz:ContinuousMove>
          <tptz:ProfileToken>{PROFILE_TOKEN}</tptz:ProfileToken>
          <tptz:Velocity>
            <tt:PanTilt x="{v['x']}" y="{v['y']}"/>
          </tptz:Velocity>
        </tptz:ContinuousMove>
      </s:Body>
    </s:Envelope>"""
    
    send_ptz_command(soap_body)
    print(f"Moving {direction}...")









def stopcamera():
    """Stops all PTZ movement."""
    soap_body = f"""<?xml version="1.0" encoding="UTF-8"?>
    <s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl">
      <s:Body>
        <tptz:Stop>
          <tptz:ProfileToken>{PROFILE_TOKEN}</tptz:ProfileToken>
          <tptz:PanTilt>true</tptz:PanTilt>
          <tptz:Zoom>true</tptz:Zoom>
        </tptz:Stop>
      </s:Body>
    </s:Envelope>"""
    
    send_ptz_command(soap_body)
    print("PTZ Stopped.")



def startIr(isOn):
    try:
        if isOn:
            set_night_vision(True)
        else:
            set_night_vision(False)
    except Exception as e:
        print(e)




imagingurl = f"http://{IP_ADDRESS}:{PORT}/onvif/imaging_service"

def set_night_vision(turn_on=True):
    global imagingurl
    VIDEO_SOURCE_TOKEN = "VideoSource_1"
    
    ir_mode = "OFF" if turn_on else "ON"
    
    # Critical: EZVIZ sometimes needs the action in the header
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': '"http://www.onvif.org/ver20/imaging/wsdl/SetImagingSettings"'
    }

    soap_body = f"""<?xml version="1.0" encoding="UTF-8"?>
    <s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" 
                xmlns:timg="http://www.onvif.org/ver20/imaging/wsdl" 
                xmlns:tt="http://www.onvif.org/ver10/schema">
      <s:Body>
        <timg:SetImagingSettings>
          <timg:VideoSourceToken>{VIDEO_SOURCE_TOKEN}</timg:VideoSourceToken>
          <timg:ImagingSettings>
            <tt:IrCutFilter>{ir_mode}</tt:IrCutFilter>
          </timg:ImagingSettings>
          <timg:ForcePersistence>true</timg:ForcePersistence>
        </timg:SetImagingSettings>
      </s:Body>
    </s:Envelope>"""
    try:
        response = requests.post(imagingurl, data=soap_body, headers=headers, auth=HTTPDigestAuth(USER, PASS), timeout=1)
        
        if response.status_code == 200:
            print(f"✅ Success! Night Vision: {ir_mode}")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
    except:
        pass
    
def getCurrentImageSettings():
    """Fetches current brightness, contrast, saturation, and sharpness."""
    VIDEO_SOURCE_TOKEN = "VideoSource_1"
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': '"http://www.onvif.org/ver20/imaging/wsdl/GetImagingSettings"'
    }
    
    soap_body = f"""<?xml version="1.0" encoding="UTF-8"?>
    <s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" xmlns:timg="http://www.onvif.org/ver20/imaging/wsdl">
      <s:Body>
        <timg:GetImagingSettings>
          <timg:VideoSourceToken>{VIDEO_SOURCE_TOKEN}</timg:VideoSourceToken>
        </timg:GetImagingSettings>
      </s:Body>
    </s:Envelope>"""
    try:
        response = requests.post(imagingurl, data=soap_body, headers=headers, auth=HTTPDigestAuth(USER, PASS), timeout=1)
        
        if response.status_code == 200:
            # Simple way to parse the XML response for values
            root = ET.fromstring(response.text)
            # Namespaces can be tricky in XML, so we find tags ignoring them
            def find_val(tag):
                for el in root.iter():
                    if tag in el.tag: return el.text
                return "N/A"

            settings = {
                "BacklightCompensation": find_val("IrCutFilter"),
                "Brightness":            find_val("Brightness"),
                "ColorSaturation":       find_val("ColorSaturation"),
                "Contrast":              find_val("Contrast"),
                "Focus":                 1.0,
                "Sharpness":             find_val("Sharpness")
            }
            return settings
        else:
            print(f"❌ Failed to get settings: {response.status_code}")
            return {
                "BacklightCompensation": 50,
                "Brightness":            50,
                "ColorSaturation":       50,
                "Contrast":              50,
                "Focus":                 1.0,
                "Sharpness":             50
            }
    except:
        return {
                "BacklightCompensation": 50,
                "Brightness":            50,
                "ColorSaturation":       50,
                "Contrast":              50,
                "Focus":                 1.0,
                "Sharpness":             50
            }



def changeimagesettings(kwargs):
    VIDEO_SOURCE_TOKEN = "VideoSource_1"
    
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': '"http://www.onvif.org/ver20/imaging/wsdl/SetImagingSettings"'
    }

    soap_body = f"""<?xml version="1.0" encoding="UTF-8"?>
    <s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" 
                xmlns:timg="http://www.onvif.org/ver20/imaging/wsdl" 
                xmlns:tt="http://www.onvif.org/ver10/schema">
      <s:Body>
        <timg:SetImagingSettings>
          <timg:VideoSourceToken>{VIDEO_SOURCE_TOKEN}</timg:VideoSourceToken>
          <timg:ImagingSettings>
            <tt:Brightness>{kwargs['Brightness']}</tt:Brightness>
            <tt:Contrast>{kwargs['Contrast']}</tt:Contrast>
            <tt:ColorSaturation>{kwargs['ColorSaturation']}</tt:ColorSaturation>
            <tt:Sharpness>{kwargs['Sharpness']}</tt:Sharpness>
          </timg:ImagingSettings>
          <timg:ForcePersistence>true</timg:ForcePersistence>
        </timg:SetImagingSettings>
      </s:Body>
    </s:Envelope>"""
    try:
        response = requests.post(imagingurl, data=soap_body, headers=headers, auth=HTTPDigestAuth(USER, PASS),timeout=1)
        print(f"Update Status: {response.status_code}")
    except:
        pass