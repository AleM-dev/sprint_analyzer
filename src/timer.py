class Time:
    @staticmethod
    def centiseconds(time,defaults):
        centi = 0
        if len(time) > 0:
            centi = 10*time[defaults['decisecond']] + time[defaults['centisecond']]
            if time[defaults['centisecond']] == 3 or time[defaults['centisecond']] == 8:
                centi += 0.34
            if time[defaults['centisecond']] == 1 or time[defaults['centisecond']] == 6:
                centi += 0.67
        return centi

    @staticmethod
    def seconds(time, defaults):
        sec = 0
        if len(time) > 0:
            sec = 10*time[defaults['second_ten']] + time[defaults['second_unit']]
        return sec

    @staticmethod
    def minutes(time, defaults):
        min = 0
        if len(time) > 0:
            min = time[defaults['minute_unit']]
        return min

    @staticmethod
    def time_to_centiseconds(time,defaults):
        return Time.centiseconds(time, defaults) + 100*Time.seconds(time, defaults) + 6000*Time.minutes(time, defaults)
    
    @staticmethod
    def centiseconds_to_minutes_string(centi):
        centi = int(centi)
        min = centi//6000
        centi = centi%6000
        sec_ten = centi//1000
        centi = centi%1000
        sec_unit = centi//100
        centi = centi%100
        dec = centi//10
        centi = centi%10
        string = str(min) + ":" + str(sec_ten) + str(sec_unit) + "." + str(dec) + str(centi)
        return string       

    @staticmethod
    def centiseconds_to_seconds_string(centi):
        centi = int(centi)
        sec_unit = centi//100
        centi = centi%100
        dec = centi//10
        centi = centi%10
        string = str(sec_unit) + "." + str(dec) + str(centi)
        return string       

    @staticmethod
    def frame_to_time_diff(f1, f2, map_frame_time, df):
        time1 = map_frame_time[f1]
        time2 = map_frame_time[f2]
        return Time.time_diff(time1, time2, df)

    @staticmethod
    def time_diff(time1, time2, df):
        t1_centi = Time.time_to_centiseconds(time1, df)
        t2_centi = Time.time_to_centiseconds(time2, df)
        diff = t1_centi - t2_centi
        return diff

    @staticmethod
    def time_diff_without_lcd(time1, time2, lines_cleared, defaults):
        t1_centi = Time.time_to_centiseconds(time1, defaults['timer'])
        t2_centi = Time.time_to_centiseconds(time2, defaults['timer'])
        diff = t1_centi - t2_centi
        if lines_cleared > 0:  
            diff -= 100*defaults['delay']['line_clear'][lines_cleared]
        diff = int(diff)

        return diff

    @staticmethod
    def frames_to_centiseconds(frames, fps):
        return int(100*frames/fps)

    @staticmethod
    def frames_to_time(frames, fps):
        time = []
        if frames > 0:
            centi = int(100*frames/fps)
            time.append(centi//6000)

            centi = centi%6000
            time.append(centi//1000)

            centi = centi%1000
            time.append(centi//100)
            
            centi = centi%100
            time.append(centi//10)

            time.append(centi%10)
        else:
            time = [0, 0, 0, 0, 0]

        return time