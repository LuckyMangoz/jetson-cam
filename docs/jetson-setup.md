# Jetson Setup Log

| Field                           | Value         |
|---------------------------------|---------------|
| Date                            | 8/27/2026     |
| JetPack version                 | 7.2.1         |
| L4T version                     | r39.2.1       |
| Firmware version at first check | 36.4.3        |
| Target storage                  | NVMe 500gb    |
| Power mode                      | 25W           |
| Hostname                        | luckymango    |
| Jetson IP address               | 192.168.0.188 |

## Notes
- USB stick for ISO (16gb+)
-  In order to grab the correct IP run: ip route get 8.8.8.8 | grep -oP 'src \K\S+'