import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

"""
Stable Vers. 3.0

Code compiled by K.C. (2025), algorithms adapted from Jenkins (2013)

DOI: https://doi.org/10.1088/0143-0807/34/3/633

Rearranged by ChatGPT.

It seems that the author of the paper indicated above does not consider the EoT effect on the SR/SS time
calculations and azi/elv angles of the sun at a given day/time combination, while those equations work very well for 
a perfect Earth and perfectly circular orbit shape. 

The EoT (equation of time) is the time difference between the apparent solar time (the "time" indicated by the earth's 
rotation -> sun's path w.r.t the observation point) and the mean solar time (averaged solar path time by using 
24hr convention). The EoT can shift the solar noon time by up to +/- 15 mins, where the standard meridian can be 
shifted by a few degrees where 4 min is equivalent to 1 meridian degrees. Therefore we need to compensate such time 
difference in terms of calculating the SR/SS times and solar azi/elv angles. NOAA's solar calculator 
https://gml.noaa.gov/grad/solcalc/ calculates those values considering the EoT effects, which matches the displayed
values of the mainstream star observation software such as Stellarium or Sky Guide.
"""

# CONSTANT List
omega = 2 * np.pi / 23.9345          # Earth's rotation rate, eq. (6) [rad/hr]
epsilon = np.radians(23.44)          # Obliquity, eq. (3)
alt_threshold = np.radians(-0.85)    # Sunrise/sunset altitude including refraction
epoch = dt.datetime(2013, 1, 1, 0, 0) # t=0 epoch
equinox_2025 = dt.datetime(2025, 3, 20, 9, 1)  # Spring equinox 2025, UTC
lambda0_2025 = 132.96                 # Reference longitude for 2025, eq. (26)
# N.B. to calculate lambda0_2025, use eq. (26):
# lambda0_2025 = (18 + (-7.16/60) - (9+1/60)) * 2pi/24 = 132.96 deg

# Helper functions
def days_since_epoch(dt_utc):
    """Return t in days since 2013-01-01 00:00Z."""
    return (dt_utc - epoch).total_seconds() / 86400

def mean_anomaly(t_days):
    """Mean anomaly M(t), eq. (13)."""
    return -0.0410 + 0.017202 * t_days

def ecliptic_longitude(t_days):
    """Ecliptic longitude phi(t), eq. (14)."""
    M = mean_anomaly(t_days)
    return -1.3411 + M + 0.0334 * np.sin(M) + 0.0003 * np.sin(2*M)

def solar_declination(t_days):
    """Solar declination delta, eq. (5)."""
    phi = ecliptic_longitude(t_days)
    return np.arcsin(np.sin(epsilon) * np.sin(phi))

def equation_of_time(day_number):
    """Approximate equation of time in minutes for day of year."""
    B = 2 * np.pi * (day_number - 81) / 364
    return 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)

def _get_spans(day_list):
    # spans = []
    # if not day_list:
    #     return spans
    # start = prev = day_list[0]
    # for d in day_list[1:]:
    #     if d == prev + 1:
    #         prev = d
    #     else:
    #         spans.append((start, prev))
    #         start = prev = d
    # spans.append((start, prev))
    # return spans
    pass

# Main functions
def compute_max_min_altitudes(lat_deg, day_number):
    """
    Compute maximum and minimum solar altitudes (deg) for a given day of year.
    Uses eq. (18): alpha_max = arcsin[sin(L + theta_prime)],
                 alpha_min = arcsin[sin(L - theta_prime)],
    where theta_prime = pi/2 - delta, and delta from eq. (5).
    """
    date = dt.datetime(equinox_2025.year, 1, 1) + dt.timedelta(days=day_number - 1)
    t = days_since_epoch(date)
    delta = solar_declination(t)
    theta_prime = np.pi/2 - delta
    lat_rad = np.radians(lat_deg)
    max_alt = np.arcsin(np.sin(lat_rad + theta_prime))
    min_alt = np.arcsin(np.sin(lat_rad - theta_prime))
    return np.degrees(max_alt), np.degrees(min_alt)


def solar_angles(lat_deg, lon_deg, dt_utc):
    """
    Compute solar elevation (rad) and geographic azimuth (deg clockwise from North)
    for given lat, lon, UTC datetime, using declination + equation-of-time.
    """
    # 1) compute solar declination, i.e. delta
    t = days_since_epoch(dt_utc)
    delta = solar_declination(t)

    # 2) equation of time (in minutes) and longitude correction (4 min per degree)
    day_number = dt_utc.timetuple().tm_yday
    eot = equation_of_time(day_number)        # in minutes
    time_offset = eot + 4 * lon_deg           # minutes

    # 3) true solar time in minutes from local midnight
    tst = dt_utc.hour * 60 + dt_utc.minute + dt_utc.second / 60 + time_offset

    # 4) hour angle H (radians), zero at solar noon, positive in the afternoon
    H = np.radians(tst / 4.0 - 180.0)

    # 5) convert latitude to radians
    L = np.radians(lat_deg)

    # 6) elevation (altitude) above horizon
    elevation = np.arcsin(
        np.sin(L) * np.sin(delta) +
        np.cos(L) * np.cos(delta) * np.cos(H)
    )

    # 7) azimuth: clockwise from North
    #    atan2 returns radians in (−π,π], with positive toward East of North
    az = np.arctan2(
        -np.sin(H),
        np.tan(delta) * np.cos(L) - np.sin(L) * np.cos(H)
    )
    azimuth = (np.degrees(az) + 360) % 360

    return np.rad2deg(elevation), azimuth



def compute_sunrise_sunset(lat_deg, lon_deg, date_local, tz_offset_hours=0):
    """
    Compute local sunrise and sunset times accounting for date, location, and timezone.
    Returns (sunrise_local, sunset_local), or (None, None) for polar day/night.
    c.f. eq. (5).
    """
    # 1) UTC midnight for the local date
    utc_midnight = dt.datetime(date_local.year, date_local.month, date_local.day) \
                  - dt.timedelta(hours=tz_offset_hours)
    # 2) declination at midnight
    t = days_since_epoch(utc_midnight)
    decl = solar_declination(t)
    L_rad = np.radians(lat_deg)

    # 3) compute the cosine of the sunrise hour angle
    cos_H0 = (np.sin(alt_threshold) - np.sin(L_rad)*np.sin(decl)) \
             / (np.cos(L_rad)*np.cos(decl))
    # polar day/night check
    if cos_H0 < -1 or cos_H0 > 1:
        return None, None

    # 4) actual hour angle at sunrise/sunset (radians)
    H0 = np.arccos(cos_H0)
    # half-day length in hours
    half_day = H0 / (2 * np.pi) * 24

    # 5) standard-meridian and longitude shift (hours)
    std_meridian = tz_offset_hours * 15
    lon_corr_hours = (std_meridian - lon_deg) / 15.0

    # 6) equation of time effect (minutes) w.r.t. hours
    day_num = utc_midnight.timetuple().tm_yday
    eot_min  = equation_of_time(day_num)
    eot_hours = eot_min / 60.0

    # 7) Solar noon w.r.t. MEAN solar time, local TZ
    solar_noon_local = 12.0 + lon_corr_hours - eot_hours

    # 8) assemble the datetimes
    base = dt.datetime(date_local.year, date_local.month, date_local.day)
    sunrise_local = base + dt.timedelta(hours=solar_noon_local - half_day)
    sunset_local  = base + dt.timedelta(hours=solar_noon_local + half_day)

    return sunrise_local, sunset_local


# Formatting output stuff
def format_timezone(offset_hours):
    sign = '+' if offset_hours >= 0 else '-'
    total_minutes = abs(int(offset_hours * 60))
    hh = total_minutes // 60
    mm = total_minutes % 60
    return f"UTC{sign}{hh:02d}:{mm:02d}"


def format_lat_lon(lat_deg, lon_deg):
    lat_dir = 'N' if lat_deg >= 0 else 'S'
    lon_dir = 'E' if lon_deg >= 0 else 'W'
    return f"{abs(lat_deg):.2f}°{lat_dir}", f"{abs(lon_deg):.2f}°{lon_dir}"

# ---------------------- Example usage ----------------------
# if __name__ == "__main__":
#     # Set location parameters, latitude in degrees, longitude in degrees,
#     # and timezone offset in hours
#     lat_deg, lon_deg, tz_offset = 51.5, -0.17, 1
#     lat_str, lon_str = format_lat_lon(lat_deg, lon_deg)
#     tz_str = format_timezone(tz_offset)
#
#     # Set a trial date and time
#     trial_local = dt.datetime(2025, 5, 30,
#                                      10, 00) # LOCAL TZ
#     trial_utc = trial_local - dt.timedelta(hours=tz_offset) # UTC
#
#     # Sunrise/Sunset
#     sr, ss = compute_sunrise_sunset(lat_deg, lon_deg,
#                                     trial_utc.date(),
#                                     tz_offset_hours=tz_offset)
#     print(f"Observation point at ({lat_str}, {lon_str}):")
#     print(f"- SR/SS time for {trial_utc.date()} at"
#           f" ({lat_str}, {lon_str}), {tz_str}:")
#     if not (sr, ss) == (None, None):
#         print(f"    - Local Sunrise: {sr.time()}; Local Sunset: {ss.time()}")
#     else:
#         print(f"    - Polar Day/Night reached!")
#
#     # Local solar alt/azimuth angles
#     alt, azi = solar_angles(lat_deg, lon_deg, trial_utc)
#     print(f"At local time {trial_local.strftime('%Y-%m-%d %H:%M')} (UTC+"
#           f"{tz_offset:02d}:00) <- {trial_utc.strftime('%Y-%m-%d %H:%M')} "
#           f"(UTC):")
#     print(f" - Solar altitude: {np.degrees(alt):.2f}°; azimuth: {azi:.2f}°")
#
#     # Max/Min altitudes
#     day_num = trial_utc.timetuple().tm_yday
#     max_alt, min_alt = compute_max_min_altitudes(lat_deg, day_num)
#     print(f"Max/Min solar altitudes on day {trial_utc.date()}: {max_alt:.2f}°,"
#           f" {min_alt:.2f}°")

    # Max altitude vs Day Number
    # days = list(range(1, 366))
    # max_alts = [compute_max_min_altitudes(lat_deg, d)[0] for d in days]
    # plt.figure(figsize=(8, 6))
    # plt.plot(days, max_alts)
    # plt.xlabel('Day Number')
    # plt.ylabel('Max Solar Altitude (°)')
    # plt.ylim(-90, 90)
    # plt.title(f'Maximum Solar Altitude vs Day Number for ({lat_str}, {lon_str}), 2025')
    # plt.show()

    # Plot Sunrise, Sunset, and Solar Noon vs Day Number with polar and EOT
    # days = list(range(1, 366))
    # sr_hrs, ss_hrs, noon_hrs = [], [], []
    # polar_days, polar_nights = [], []
    # for d in days:
    #     date_d = (dt.datetime(2025, 1, 1) +
    #               dt.timedelta(days=d - 1))
    #     sr_d, ss_d = compute_sunrise_sunset(lat_deg, lon_deg, date_d,
    #                                         tz_offset_hours=tz_offset)
    #     if sr_d is None and ss_d is None:
    #         # Distinguish polar day/night via declination vs latitude
    #         decl = solar_declination(
    #             days_since_epoch(date_d + dt.timedelta(hours=12)))
    #         if (lat_deg >= 0 and decl >= 0) or (lat_deg < 0 and decl < 0):
    #             polar_days.append(d)
    #         else:
    #             polar_nights.append(d)
    #         sr_hrs.append(np.nan)
    #         ss_hrs.append(np.nan)
    #         noon_hrs.append(np.nan)
    #     else:
    #         # Calculate sunrise and sunset hours
    #         sr_hour = sr_d.hour + sr_d.minute / 60 + sr_d.second / 3600
    #         ss_hour = ss_d.hour + ss_d.minute / 60 + ss_d.second / 3600
    #         sr_hrs.append(sr_hour)
    #         ss_hrs.append(ss_hour)
    #         # Solar noon: midpoint plus equation of time correction
    #         mean_noon = sr_d + (ss_d - sr_d) / 2
    #         noon_hrs.append(mean_noon.hour + mean_noon.minute / 60 + mean_noon.second / 3600)
    #         # eot_minutes = equation_of_time(d)
    #         # solar_noon = mean_noon + dt.timedelta(minutes=eot_minutes)
    #         # noon_hrs.append(
    #         #     solar_noon.hour + solar_noon.minute / 60 + solar_noon.second / 3600)
    #
    # plt.figure(figsize=(8, 6))
    # plt.plot(days, sr_hrs, label='Sunrise')
    # plt.plot(days, ss_hrs, label='Sunset')
    # plt.plot(days, noon_hrs, label='Solar Noon')
    # # Shade polar periods
    # for start, end in _get_spans(polar_days):
    #     plt.axvspan(start, end, color='gold', alpha=0.3,
    #                 label='Polar Day' if start == polar_days[0] else '')
    # for start, end in _get_spans(polar_nights):
    #     plt.axvspan(start, end, color='navy', alpha=0.3,
    #                 label='Polar Night' if start == polar_nights[0] else '')
    #
    # plt.xlabel('Day Number')
    # plt.ylabel('Local Time (hours)')
    # plt.ylim(0, 24)
    # plt.title(
    #     f'Sunrise, Sunset, and Solar Noon vs Day Number \nat '
    #     f'({lat_str}, {lon_str}), 2025')
    # plt.legend(loc='upper right')
    # plt.tight_layout()
    # plt.show()
    #
    # # Analemma plot remains unchanged
    # analemma_hr = 12.66
    # dates = [dt.datetime(2025, 1, 1) +
    #          dt.timedelta(days=i, hours=analemma_hr)
    #          for i in range(365)]
    # elevs, azs = zip(*(solar_angles(lat_deg, lon_deg, d) for d in dates))
    # hours = int(analemma_hr)
    # minutes = int((analemma_hr - hours) * 60)
    # time_str = f"{hours:02d}:{minutes:02d} UTC"
    #
    # plt.figure(figsize=(8, 6))
    # plt.scatter(azs, np.degrees(elevs), s=10, marker='x')
    # plt.xlabel('Azimuth (deg)')
    # plt.ylabel('Solar Altitude (deg)')
    # plt.title(f'Analemma plot for ({lat_str}, {lon_str}) at {time_str}, 2025')
    # plt.show()
