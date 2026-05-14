from onvif import ONVIFCamera


CAMERA_IP = '192.168.1.100' 
CAMERA_PORT = 80            
USERNAME = 'admin'

PASSWORD = '123456789!' 

def changeimagesettings(kwargs):
    cam = ONVIFCamera(CAMERA_IP, CAMERA_PORT, USERNAME, PASSWORD)

    media = cam.create_media_service()
    imaging = cam.create_imaging_service()

    profile = media.GetProfiles()[0]

    video_source_token = profile.VideoSourceConfiguration.SourceToken

    req = imaging.create_type('SetImagingSettings')
    req.VideoSourceToken = video_source_token
    del kwargs['BacklightCompensation']
    del kwargs['Focus']
    print(kwargs)
    req.ImagingSettings = kwargs

    req.ForcePersistence = True

    imaging.SetImagingSettings(req)

def getCurrentImageSettings():
    return {
            "BacklightCompensation": "OFF",
            "Brightness":            50.0,
            "ColorSaturation":       50.0,
            "Contrast":              50.0,
            "Focus":                 1.0,
            "Sharpness":             50.0,
        }
    cam = ONVIFCamera(CAMERA_IP, CAMERA_PORT, USERNAME, PASSWORD)

    media = cam.create_media_service()
    imaging = cam.create_imaging_service()

    profile = media.GetProfiles()[0]

    video_source_token = profile.VideoSourceConfiguration.SourceToken

    req = imaging.create_type('GetImagingSettings')
    req.VideoSourceToken = video_source_token

    settings = imaging.GetImagingSettings(req)
    settings['BacklightCompensation']=settings['BacklightCompensation']['Mode']
    settings['Focus']=settings['Focus']['DefaultSpeed']
    return settings

ptz=None
ptz_profile_token=None
def movecamera(key: str):
    global ptz
    global ptz_profile_token
    mycam = ONVIFCamera(CAMERA_IP, CAMERA_PORT, USERNAME, PASSWORD)
    ptz = mycam.create_ptz_service()
    media = mycam.create_media_service()
    
    # Get the first media profile
    profile = media.GetProfiles()[0]
    ptz_profile_token = profile.token

    # Create a move request template
    request = ptz.create_type('ContinuousMove')
    request.ProfileToken = ptz_profile_token
    SPEED=1
    if key == "up":
        x,y=(0, SPEED)
    elif key == "down":
        x,y=(0, -SPEED)
    elif key == "left":
        x,y=(-SPEED, 0)
    elif key == "right":
        x,y=(SPEED, 0)
    move_request = {
            'ProfileToken': ptz_profile_token,
            'Velocity': {
                'PanTilt': {
                    'x': x,
                    'y': y,
                    'space': 'http://www.onvif.org/ver10/tptz/PanTiltSpaces/VelocityGenericSpace'
                }
            }
        }
    ptz.ContinuousMove(move_request)

def stopcamera():
    global ptz
    global ptz_profile_token
    if ptz is not None:
        ptz.Stop({'ProfileToken': ptz_profile_token})

def startIr(isOn):
    if isOn:
        control_ezviz_camera(CAMERA_IP, CAMERA_PORT, USERNAME, PASSWORD, night_vision=True)
        print("IR on")
    else:
        control_ezviz_camera(CAMERA_IP, CAMERA_PORT, USERNAME, PASSWORD, night_vision=False)
        print("IR Off")





def control_ezviz_camera(ip, port, user, password, night_vision=None):
    
    mycam = ONVIFCamera(ip, port, user, password)

    
    media_service = mycam.create_media_service()
    profiles = media_service.GetProfiles()
    
    if not profiles:
        print("No media profiles found on this camera.")
        return
        
    video_source_token = profiles[0].VideoSourceConfiguration.SourceToken
    profile_token = profiles[0].token

    # 3. Control Night Vision (IR Cut Filter)
    if night_vision is not None:
        imaging_service = mycam.create_imaging_service()
        
        # Map user-friendly boolean to ONVIF IrCutFilter modes
        
        if night_vision is True:
            ir_mode = "OFF" # Filter OFF = Night Vision ON
        else:
            ir_mode = "ON"  # Filter ON = Night Vision OFF
            
        print(f"Fetching current imaging settings to apply IR mode: {ir_mode}...")
        
        # Get current settings
        settings = imaging_service.GetImagingSettings(video_source_token)
        
        # Update the IR Cut Filter property
        settings.IrCutFilter = ir_mode
        
        # EZVIZ cameras sometimes return unparseable Extension data. 
        # Clearing it out prevents Zeep XML validation errors when sending settings back.
        settings.Extension = None 
        
        # Create the request
        request = imaging_service.create_type('SetImagingSettings')
        request.VideoSourceToken = video_source_token
        request.ImagingSettings = settings
        request.ForcePersistence = True # Save settings after camera reboot
        
        try:
            imaging_service.SetImagingSettings(request)
            print(f"✅ Night Vision successfully set to: {night_vision} (Filter: {ir_mode})")
        except Exception as e:
            print(f"❌ Failed to set Night Vision. Error: {e}")

    
