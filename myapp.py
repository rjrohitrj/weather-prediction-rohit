from flask import Flask
import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import datetime
import os

server = Flask(__name__)

# Reading the dataset
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, 'Climate_data.csv')
dataset = pd.read_csv(my_file)

bg= ['https://lh3.googleusercontent.com/warecodOgobhPNUVauAZ8EWIZEwzbKuJsWC0yYiVi1BCldx2RcdJBhxdUvX_2P-So-wWM_hlEhULXVgqDVSCFC16acIwhQiFt-n8Ti1K0SO1_JUY34GG_Z-rtvYDkkvGfugnpM4Sx2j9zZl43qn2dC09z4AV1c1FiKEK9gvbGfFGhcmPqhios0Q7evSxmdrsnHdDSOG4le0dQlNpI3q2FKNp_QZZWNExb0B6P-X2EbR9uG1CxWfxWo-26bCnOZWmjoIVll4gKnVneJkidnBych20mxof1s6jKccbdcFkWUsRuL5foSAeT3jTd6dqdUOBTrGr0b-8W3nxbKQPmNT76zlPoMuDW6WmBRS1c0dSJPR4l00HUFEGTgn0_m6Qy_KVKnLRVcH7b5qrvnr246cW8ohuTmWZUGPqQLRwfI8GCTMsFvM5yXuiQ0vpnWzbbwQBhtS55FC4ifeNO2XF1mO-4-p65BnIuwi7SdwvN51xWD2s_DuprW2ko8PntUIKCvgzHPKdhGoRMdx2VuB_7_vmygZUeMD8bKJz21Xa8sS-o2JXb8_cmuKioGUO9Ibz4yTW5Q5J38Ty2hWNHO9rCy0465Xngzmsb_imXl62u5ncbZfHwleaoY7IitMtYGuxdr4YgOCbBIJjfJnVzrMNPhT5Wya85w_toTdv=w1177-h662-no',
     'https://lh3.googleusercontent.com/kWA7KeKf1-JGXXdjLCEE3hxfN8-MqxIeSFinEl2EbmO8DUHdG9lc18bhcqn7bPR_USGkmhnMxYXCWvfr1kqPlYc6ZIqS7jB0GCJp0maX-q5CuRi0JBw5S7d8QRnKJefHC4DnhW8rKGgoqoegNOcQ9nQ-0LtCGT2NNhq2hMQ3fd8xXE0eUr3gsK-nVqOnhOP6yVfbt58ZmejV3CyUp-eAWOfV9kj8ttl9wsnxgi5lHwSNWzFu9ZJ20Y8W0g46pcLkRDcpI1Hfu73IYTPWKQr62lIAFgmeOgrQQSG6LdkTmncNMz5HbU4MG9Z79Njq6e-wGRzSTIBuJVnGS-zoWkErzw5YjcASkEUTTbCwNtgW7D6xQ96SnfXC-YrZzIqVr3zFJl1ShG_L68uVwXA68IQ-Oqu1R1W9MurAFhg8vTwyEZ63rd5n2oOkpBHE3dOFkK62BwnmBbtvt2wyPSVeHy7x4GtSm85Slz8LNN2iUkNgYbHpH0bT33-w9xt3-XWlgSQsSrYwCwkTxm8ycvcY1Ge42YgI7jVKAm_NOaAKozWh3AuIYk3sqJx7P5-_Ij1hCCGNaGDB9--0qCysI7X-Bdsay76M68UQy6cBjx0t0X7gSeBxyBEICi7ZJMK7DYGfTmNu5sB9RwvusWEjx0Cinj57B6Ez82ES6-iz=w1177-h662-no',
     'https://lh3.googleusercontent.com/OTwrx_MK1_DMY_cmPV-iAt3BVD98JDNGiyIScQxUHvQrO5wsSZZBXInduKLmnpnqBX9urVb9gGw43tMpBlyG0MQOMduOYiImccv_XI8Q1JxZD8iaVerNdLLBMI73BTwbhuZ5B0KbTVMpyMHHYwhJXjm1Iui_IP424Eg9eJSJj2i-dbKLaXfAq_UWQ0ZZvaNskje2CFtG2qX7L-eg9YPAqn2efabm4S4wZWHfUWZqDUyr-wHrD_GJKHri6ALW6-vM6FqGndW-DM9ZitVO2rQogRsVt_tPUr1YHajxLL5WKMEInW5wyHP1b0Dxh-mmXmY-qycqeTTPEQLgwOf7okNd0tnpos2UtaiqVDDwsl00Ksta0gpwBaOyTR-SLtlhogo83qwts_rveS741AhQ0swFDbNLV2SOj52EEu3eTmqpdJGdA5NcaYFOsJQsS6AU5oRROr2VAQ9igGGUsgWc0KyAjNPjaqlZKisGQwnGJzX8QGJNZR_n-uaA97upszVbhalWYitHJMyKvtWtB7c-4pwaGfgxrmV9ahSYI6bXwwdWrb4Z9mqLx3b6vocnhtDtAgIsMp0rrWE7a8bQzL8KH9OXy-nKt5dinQi46fmOtlGKWQ0NLew9lPnLcjsJPK5e74XWm2-2QwZJeT6XiBQKfQfUAjNxPgM32KWV=w1060-h662-no',
     'https://lh3.googleusercontent.com/FxVoo7rdDaAVkeSc7C7X3JsJfooG0bvhxi75UpPUSDbFtQiIeZ8snhABiLD2KsU2t8t0PazTESWYRswUc-haH9Qf5v7MJffxVF-jzcKRQJBrbNzWmU3hbgnF_tN_dK2sukl_6gwT9pYIWH-9J9Kh5sT6QsdoAY2ng4MvyJkn5rZZJmDMmMrTlegFyvqh_P9hiznEu9mMwk_AtCa6per1npIsJLTbKnn-dzg6fRx886EDzDgAAdEJmPkpRfTP3ODeAeKyuIBX39qd2wff7_Xm0EPhooGD8ZHPM2z2lGYnSrjtTEaEMpSOtMAcDYjZmV-Ak4uE5PvQD0iT4XgemLhvxZqq__z2H5smyqKV3WIEnZGci18-N_BSomj1jcvB7SW5wFRjdIiaCCwlakm1ICZzNT3rOVk6AqcRaiASrSRII72ZhmEIoj_oWuKgEtzR3JHmpBf1hdEdRu8x-gsuAkkhvBS89T-aznDQg_DC6oe9pRG14BzzyMYR0-hQlQF3WPgiZII79MO0m4fU2goG2SvUPR06VL-ETSYrHj38FMVyYkIRrLghe18eUdVyJtXfiHicexnjVQUuBLuxiS7TK8KA3yPyDElnjADgZ24WgM3g3V42jpg01PJIehbx0zRMhQ0Gq3q5Y3Khxj3Jia0-ALBUZrB2tCr69cgv=w430-h286-no',
     'https://lh3.googleusercontent.com/fmsD8SFZhr3KVYcVu7_POjIcjouE0sAupdiOBsaHtZu3HeCqu561znF1Oa4VnhRgGNbw4Dp9ZGV3_9-d-lNDvoJLvmfmofKYBasSihyKE7f70xsyFppuz7m6oKJlqmyOXM-K8n2SjapRyKNPP83avxBVdgYz3Qz_rmsfTKhSPIY6Clv0S6IXOZmPzZxUl-sbp7plBabr8vr5nuwrvK9Lrn52VvRZrWlLW__OPVkAMJCbmWAaymi-z__aNAYEc1-CelQ2NSY_ZxJv0OWPKcY9iaXKjmElfy_adjc1I_ZTfy6jdYE2Vwmyzxckelw6ZK_QIJ8NQg_GnNr36YWXNx6FzhzToAFsGJCrMEUJj7u_KEzaOfXJt6gdsZyg7y1NwPNFErS7Aeg_x1Bmd_d4E-SZNx5f4UR4w3_ZN1TL6XmYVB_rKxjaS6eMMz5oX-KTyECYJ3t_YUX7zx3dpMBxJwSiV769rwi3BkYbf2Py1QSzqu_hsSZ4iZqZ5QJvyZnkKY20Trl1Qomcd8jksBn9kA2hANnTbDRdTPhphxhY5MPViypMCFOiRZyvmJv85w-EEMVzAEs_Pn_8dEgFmqOOMgclrOJUqv7BWLdfGJmAlk1JAIQFseYHlFNSsp-fb4D6Wf-e-MOHor0y3r5Ve_HShRa0NmK3H6Kx56mo=w975-h662-no',
     'https://lh3.googleusercontent.com/vZRmVtyimeSNX23i2eq0qqkVe4QH_GLtmg5lPStj_j5-2rA7YQfv1ysNRRoQUQ9Uahl6UmaN9nh7DKh4Senx83Us5l1K3vagSiZMTLpJPETuMfKSpAuVlAoVBgz4o6fwTDNoPdoGk0X3NANlTIaw3hqnuUoqyocYQA_bT5VJC5qeC7mXFpKpb-z3Pw6-bOB82iDhwymx77_WIxwUeqfHG8UtI0rTKUVYjELeLVBgJmngDNBnVz6NmB35HBU2nMYza_xryuxiEH9Z8d09x3zZJ70xq7yYAqjfA2Chlsq185gqthKZngAT0dQbVH9KEWmGiQPqusA8dOedoTadz8dgmISRSJYIJsNpSgjn3oeDKlvrURrRuOGsDuZgL0HDKbEZRLqIM_OYxT0vt9zbex9I--SqO113zWlZEYS3xKAHsIPCfHUplZERWuJPboxdmswiYPPezutxNbMcYYxzkH3a00d6ogMrFUlK1t04qR5jWe-c62zqe-PM1uEazk7UdPdDo1po3IbMAFAya5epErmY_yKwaVndncx_8jvxTmpIIjC_Cz1ZGsd92JMM5-WY1rbHFOgpym66ln7yhNAVF3CF_4PYFmXC3zOku8DicxkS8M22YRNsIkJ8s1n0aHKR5B7EGXnsTXYcO5LqPySsAgHKeS8DgMvupnch=w995-h662-no',
     'https://lh3.googleusercontent.com/xE1CjQ7Ro0GsTyCcbDpjiR44L6BtlaLgWAWs48ns5g7hdG-OXyRhgRkXhU5X-nlsgO_dqnXZtikLPrqNODSf8Son6Kd9BHZHYOOYwWhUqWsDC4Bu0cRngj68ptdPcjQe-ckMu7MHIJUdAYCvz0UDVjTnlU4GostxFzaiGSvcMV5URc_6lkQOwhHM7OV07Gg-G6uwKQBb4Qwkj4d2wIhIvNg85PWYd8u7wCdorFzxehN_sO_Tk20CDNCwpsgTTeU6mIbiu9nyfgW4ZN-w7xoFfN1PgyARf-DoNSaDoDoRup1ZXjfMZksicROMkXS4rRyXZ7UqePVtiHRnjlDc_lvMDLr66UVaEntz3zrSN29ZKux6Qg8QJRq7wsdDEqe_soihPh98fE7iSCmOh1i-aPEG9_a4ZZ1WuH4ibQXf1FJslaBULe1AH8wS_2H623kQBZoiEFMOvy-IE528MdIQc51m1wPO9NAkM0UPnaEB0Y2EKLBph66OHbygvNEg_T7wrLO3QXOeaT9oVMzpuKsEG13GohKOL2lFThRFWY4XGYKwyddQJnIKZKvoUnJ-wWWCdLgkyXn9Ug59Yx3kLEY5infwqTZa8WT4abc_QYZCBsYXCKTtk8zL_4E0td-G4dBwfoZplsHVdYrt84yFGkxvgdWDepyQ1BiZ5xhp=w993-h662-no',
     'https://lh3.googleusercontent.com/On4JNJxlPBU0L1VLId5CUzX0zjUwicu_wovV8QtQFKSG2u9cw_xXUZFQ96UZkeAmuWI7qN6Z4p0reOof564xcLSQZA8lWHmbeSdGu1IUDgml_dak01Cl58Wj3QQLTrWS4eiHawsFAvGTC73YwlQ7li1JoLXwJsBIExCOUiUhT6lRLHHgs4zhNUUbamAyGWXMYexafdx2FCRUvfRJ3TyQaRDYpW1EOm-NU-ZYSwchjaq_g_QU4nRXwkBIltIMpsv6eB_lx6GMa_5DHBKRYRzco81AUPF_ovmf1Ra4QmA1K_aoAh2oq7axi8sFLYGqJuC-S7KRlHB969a65o0dYdfyffYnAABItL5wYCW7uCt49wnl6FoRmSHXr8uqyrxiCzdYT8EnXGlVFImZAz4yOxwmmOnlwLRqoBnguREQLlgtK0e-gBTZxLeHe3rgeG2Uc4UAbEdYHW5rtXvZsm0VPaRksWMAvAamasz56EmLzoyoaQIpI77NSP6lF2RvnvOFCGI4YGEM5r0XnMSorVMnLNytq66uORN6SkdQOEAiGbyyLa8GwfWMDN-Eo865_yyF3bVVhEMbB6rl4cAlE3MMgylHpDaGXFaCXEq05oMbjyE2w6reSDUG4DwL1PAbPbN-TGVbFuA3whECe67Io04TeTcerraePJiN4UA6=w657-h369-no',
     'https://lh3.googleusercontent.com/Df0hP31F7m-Lbh2UETZ30g37j7CAM9F0eHndApB-xiyVbjRIqr6WSQQYFU1kVU4awlvdUwx3c5pjCKBx2O4mDujrAzz-vFdQdeO0S3IokAx5IVXLM2jbrOYQBIAE2TptM74YKFDN0H1AGlQubeYu3UILF8DUl8_m6W_voHb6bUzuqmk5ZGP7mlDZNjUFTxOYdLwOKNdrCNg5UgwqNDmjDkxv9oR2HotvgasU-BSTV8BxAPauDt3ATGgPgPzEBfDaJmvLi4XXJNVANAxD8x28FD31U1k1vBMSoxkZs3VlIiYIsoe0v1H_xt1mbpWS37W4g-GYOxGQKh2a4NMCDo6i82z11yWGEbdHms_wCWxdIrtMUxCHHewrV3KbGHtJkyWUehCjbSBtdxNVW0Ep_kzs2WlJpEyCnmXzSX5YmkSjqFRn6OmoFb-v8fMcdCEtFcMieijrhIK4IB8kK3PkXJ49hNu6UJgI3051qIfXo1gqGtNphG60a6xAIysT6xG66jLHzJPiORkM1ROkG_sCY0dUS6WUHHKXdWKxl-L5joT2G3Du4aqgoAoy343mG4u8aPFD7nhl0Vrs2ZX0xnPCgMnsbgrvbf2ePzXEZ7-7dQjXF7aSanhMkZ1mkR1WXI_UvYunDhPQg8wY5MCiyRx8a2RNnBNLFmWeyMAs=w1179-h662-no',
     'https://lh3.googleusercontent.com/GzzdlI8lAfyBxQVemgpWIqpVm7DED22Tj6TKEXu084RPVy-qiRlV8XHZuwnc25T-W-IvdXjuqT-nUPBmeOvv-XX4u3LhSytlq68LzOhxfg4dX9fL1AWssObRihEXZ6lXiiyQ0u9px0G7vZZri3wsUE-TJkqOQmEfOL7JfBPXgJnHCN_Z14DIRRF7rzK2kaafp2Y2W2boEXKjU2XmulSWfVKagCM82rp7oQR54d-V8I7xTGu6cTgGa3fEB6GRYfm4x5foi3Hy46c2ubVRScKht11qzKP_unHPwaDetyhdXlHka1OIp0-OqQ9MsXorD3_4uGxa_WM3KmXkWszS8mFnOM9XdMl1UgZ2zLaDpveXT7pavvC47lvHk1vc38gEUBXIjy2a5bhYOyVckw-0n848i8rlN-B1SeKQ83-IBtrNkQ8yF5nD_nS2J-dlJXptBHNyLJ7XltYN3Dhq-FaL_hZHWaRvYA17oFhVX8uud0eTS8x3K-TUg_svx8SOp0Utg8hB64YCYdUrI80eLH1ZA7gHerVcRfoCTBN2nZE8I77-o8AmrluBmuDMo5zyIzaNr-iU6pnpYm_XLcy6OkCVmA8N2RyxCTPcHHCJJ-SZLdUJrOzlk7I5X8N0ltOCASRGr1s3EMq8MYV-vwgEwPQof7my0rQ64LfG48Q6=w412-h274-no',
     'https://lh3.googleusercontent.com/fK8ryp1EKTBqX52ronhEJfb5phoZmIcXuruC9kpeLZ3fpEyUopjCfJ62Fy3LAlGct0UKB3PTgzr1m3-WCzjmgDSWKR2woJZhcZSRmZcLQO0BMX4X7tNWrF-CkRNzs6OvGdGDQTCm3BgK0EmcgIV7zDNwuWEoTkyqTkHzVhjB5393jYfPwIDw2CIwB3oRn_BS2FGZLknwMtKBdjrbQcewiuZKCYHGxQOYJ3Vg3XJtSREZ0rHikTshrWQpKVsRxpum-GvQjhsoDTsUW9bIwYETAPOlLURLVi_trLtNakkOh9bu3aMtpfs2GUm7Yli8V3IayTxEc-TAoEaEbYugxuvmMQWqaO76LQ-mjWTddtG365HSBwjW-jw9PZjcvl9Rx6NvW2Bza5BT8yXvzGxyXlMJB_93Kcy2mfieK0v7-DzZgU7uJpT5eg3H8QqSQFgDFP_wxn5nX2e2oMvrkOtmyUdimcMf7DF0sMTD70zEOUcFZdKx2-HP6vlz9wWy44hAaCRdxclL9PPoBMZZ4CSXHvyHmtZ8y7kohXDj5f_J_oOaR8mYw7vMkbuDYwo3vIfw_Kvn5PBORE5m8GDL-8fJA_R8nJMVivboOiT6qbZeiv6geD8IIEx8elt1D3q4NA2aMcEYlxALO97eAPYeJKxNWJ6Fu_SOBSPOP_lc=w411-h274-no',
     'https://lh3.googleusercontent.com/Mgjkavt60FH5UkwHw53YxXmSRduP4Q_Zt1gBPOiJSd-7KWoX7APDmEAf0cGB2CgWB7o7L1pr01IBuZ0mOsEamJhOf5w7e9vZKaIO-fDXBkP0tq6FphoVNOeNkwWtECvYCOSvVVvJsqtrfrO42kXsY62DCJT_Tu1XOPxAsD4mKPw24tGlGQ-9lTs2Ia8Ega0jQeZgfYCmu2EPC_0ZRGg_N_dv2Wb24bJiO5oKuZmOWWlS-BzcEp03DVXtDgtdL1CL4zYjT90JgLMS_EiTfAF2K50hUIsc2fuQUKSny7TaopKTUVAldmUti2VfIgXhE4qFUcOeWn961oeui0l8BLWRhRzbmMYchKYfWoy9iIFoHjYywds9JCNGw4nJz2shwl-5ye426a4SwB1ohLdG4UZNy_s77hWxhc7ldK_6EFu7FezbJFsiJqnmXYZ8-JNKEU5wRxOp_OR9PkxGTnHlkZRBgGZznEiaySswVRP_SsSDOFrZ-VoqLkWVpJIKYqcHJ5vVpJ9UL5LfQdP47lFI74YI1_YYJCLjwe2tWBbM3hsXCnfJ93nF0BuD92f4Bcgihw7ARgtgULoazrW4gDoh7QkRrbm4FDqm7e-xZezZ_gE0EU3Bp_lbluyGFfv6-7-Mm8JVmoXbvxQ1DIfzNuU7jexhWQpLb5HpRr9-=w492-h277-no',
     'https://lh3.googleusercontent.com/hNGeLAo3i98Vo_Zb0pcDlrWMmvKrtDxwzdMI_z_COdjP5o0IoVs8J2sRHFIYzUchF07wPzVQe7Hhq6VKTUmxW8kT4q5qX1PXli9LhPk5zyJhrPCCf3iLCexfud259Endu4uzOAZCh5AwDH83HI88in9hz9FBzLbiQoNIj2y3kAX-lSCknTeEEyRYYxi2wU_jNp6QyLcoWqcojCHEiBA6XmJhPyQktyf60x4rOhgjS6Ys5n3HpOG14QrmigR18f0oJi6jDUL6dSXXm6uqpITeOt3kGIpJ67KiDy0bBfLWxYTlBdF0YwWI6w4h7FS1GEK8NL1mSxyWHeqqsDKLDsuyZj4wJFVl3hVUz9V8tT5rtLDgrsuDLs4UlkafwlFrpLfhhfVnOD-uo1N5Jv8g6cqoBOxGXkO4dbGRozBfvSGCsWNsb2ADLP0ThPy026SEoPXA0-kZxNSQGxURlvds8UYa6bPOE937Uibg7Crt0JLUX98pPmVC6aMHvbGlOGGvP-fbci9c-xHasJXHlpCswtEOkbUvzRqLMPlzRT9zSgC-t4ROy5ZMpMRN7wEO5yGp0hVXbkT7AurKmB19UnDfS_w7H5nficVi3DN5ptII0p0jbniuAk9vAhxygJVhfHFEvA1XFNEIflMsaa49YHnXaUPbDJCOWleFI9pn=w995-h662-no',
     'https://lh3.googleusercontent.com/XSVD4uO-a1u9RnDkPEQDh_1DykErUZJ2ycf-iAYZ-iz1mInjvtmqfsyNqnrJoPYVtPIgW9GplddnWnyTuLbzE96Mf-ePd3XQxSzIX235veGmfVFcMmu3QnBubq9CLtC3qlzjfVZjPkIuiYv-0HjVPicdDXCFCeqlI0kSkDhhRsH9FruWJGpiKShmKkVBY4z1QKkJkyLRWIGcFkma2k1Whcw-oJPnjmKxRM3aoZCN-sPq8JejlyCwBNW91TOA5VCCdA4xjK7Ss4kHgV_psSs6CuHSgjtYCFYzxjL1ttCwtCqNyJv9qatyyKlVseA0kWGEKZdah6Wt-t4gUDkGwMttJXvoB7pMm5mlWBuWeVSX0sDZrtUrtiBS006FY8D3Il5FKu2V5S59rKHPKfn0aPELllUWT5B40Rc9b68z6MTe-1Ga_2QVMpv1lfpufWvR4-P2FdD3PZyh_CV5-hJgBybpO-pr9UYGza0ywQPVyqMzIet9guqJsUWYG_7AQuSd44NFEGgiMdx4G4MIc7JdXaBeg3v9gbWIUS8b9dRZsK0sD6V8YuMLxW9V6Qn5PcH_4RWKf5DX62ClGfXySbwLB-Az7kTlZTP9gaSrkRThdunKf99M84VPT5OTOYNwhvnofQ5ybaJ5a0y8279vmzeLcqD76GkB6VVvdHPt=w401-h277-no',
    ]

team_member = 'https://lh3.googleusercontent.com/DP3wtOXb4ZUb4DjDDIDVVDXsPHk15V0egKDJyAniqvcHjOXT9p-dS8nSfMqoN-naeAgMwRrsJVCWDZuXdnpL3rWY_CpOVfFAFEr16XyC1fZSHFYC7t0K_irO-PrbguvL-dkt1Vyrvqn1VDOTosT9pXpT6Oct0AFXPD0P6J5l_xceKiYRDsmmjIeizGCUpHqa5dfa0X_3SAze7lMtuBAhXGpz7FL2RUpQn14SNDlIryBBtP0SoKuneywMqOMLbBfacrIraqumJ8S9XIhwrK5qa_X3hloRtSPUGyIFY0zuaiIfhOfCowSO_WUekhc31iLkv0cQIsOepXfmzelqr0kkzWXPHtoSesU3qozhiJwkfj0Ro18Svwm-2ONU-3jJmJvM5fgKkoIlJYRKNQybOfd7DNvUF2u-pFY9IvOib_xf2wHb60jLtyGNLbNo1apA_i2Oo3y-m111iIBrc9dlbVLpC62kYvhj69trD6YzGOXR8rwOKYwXnh-6iw4Ua-qxQ3mufTRbyCZgWsDxpCQFaCRMGvfWe7xjQ1vp4hYwxfDmNx7HH5_Rx92-eDioSJoNJ4hSXkj_n_5ebIenYM5QShlDha3g-gajYseBEPi1WPH_QR58ENQTFXQjVqQM6CtrAGFv-4egAd9r3o6Z3srHCZiJbhFhyOwqrq6I=w1345-h530-no'

# Style for Html Table cell
def cell_style(value):
    style = {
                'fontSize': '2vw',
                'fontFamily': 'Comic San MS',
                'text-align': 'right',
            }
    if value == -1:
        style['backgroundColor'] = '#9daea4'
        style['text-align'] = 'center'
        style['fontSize']='3vw'
        style['color']= '#33404d'
        return style
    if value%2 == 0:
        style['backgroundColor'] = '#c9e9cd'
        style['color']= '#7e011e'
    else:
        style['backgroundColor'] = '#81cb8a'
        style['color']= '#e30336'
    return style

# function for creating HTML TABLE
def generate_table(dataframe):
    rows = []
    for i in range(len(dataframe)):
        row = []
        for col in dataframe.columns:
            value = dataframe.iloc[i][col]
            style = cell_style(len(rows))
            row.append(html.Td(value))
        rows.append(html.Tr(row, style=style))

    return html.Table(
        # Header
        [html.Tr([html.Th(col,style=cell_style(-1)) for col in dataframe.columns])] +
        # Body
        rows)

# app initialization
myapp = dash.Dash(__name__, server=server)

# Setting layout for web page
myapp.layout = html.Div(id='body', children =[

    # Date Picker Componnets div
    html.Div([dcc.DatePickerSingle(id='date-picker',
                                   date=datetime.datetime.now(),
                                   min_date_allowed=datetime.datetime(1996, 11, 1),
                                   display_format='DD MMM, YYYY',
                                   placeholder='Select a date',
                                   reopen_calendar_on_clear=True)],
             style={'padding-top':'5%', 'margin-left':'5%'}),

    html.Div([# Table for Predicted Temperature div
              html.Div(id='temp_table', style={'width': '25%', 'display': 'inline-block',
                                               'margin-left': '5%', 'margin-bottom':'8%'}),

              # Blank
              html.Div(id='blank', style={'width':'25%', 'display':'inline-block'}),

              # Graph of Weekly's Temperature Variation div
              html.Div(id='Graph', style={'width': '40%', 'display': 'inline-block'})]),

    html.Div(id='member', style={'background-image': 'url({})'.format(team_member),
                                 'background-repeat':'no-repeat',
                                 'background-size':'100% 100%',
                                 'width':'100%','height':500,
                                 'margin-top':'5%'}),

    # Dash Component for Changing Background
    dcc.Interval(id='interval-component', interval=1*4000, n_intervals=0)
])

# callback function for date picker object to predict temperature and showing Html Table
@myapp.callback(Output('temp_table', 'children'), [Input('date-picker', 'date')])
def update_output(date):
    if date is not None:
        year, month, day = map(int,[date[:4], date[5:7], date[8:10]])
        selected = datetime.date(year, month, day)
        d = [selected + datetime.timedelta(days=i) for i in range(4)]
        dates = ["{:%b %d, %A }".format(d[i]).ljust(18,' ') for i in range(4)]
        temp = [str(model_training(d[i]))+'° C' for i in range(4)]
        df = pd.DataFrame({'Date':dates, 'Temperature':temp})

        # returning a Html Table of predicted temperature
        return generate_table(df)

# callback function for Graph
@myapp.callback(Output('Graph', 'children'), [Input('date-picker', 'date')])
def update_graph(date):
    if date is not None:
        year, month, day = map(int,[date[:4], date[5:7], date[8:10]])
        d = [datetime.date(year,month,day) + datetime.timedelta(days=i) for i in range(-14,15,7)]
        temp =[model_training(i) for i in d ]
        return dcc.Graph(id='example-graph',
                         figure={'data':[{'x': d, 'y': temp, 'type': 'line'}],
                                 'layout': {'title':  'WEEKLY TEMPERATURE VARIATION',
                                            'plot_bgcolor': '#DBEC9A',
                                            'paper_bgcolor': '#E9F3C2',
                                            'font':{ 'color': '#0041f2'},
                                            'xaxis':{'title': 'DATE'},
                                            'yaxis':{'title':'TEMPERATURE'}
                                           }
                                }
                        )


# Callback function for changing Image
@myapp.callback(Output('body', 'style'), [Input('interval-component', 'n_intervals')])
def update_bg_live(n):
    return {'background-image': 'url({})'.format(bg[n%14]),
            'background-repeat':'no-repeat',
            'background-size':'100% 100%',
            'height':650}

# Training the model for weather predictions
def model_training(date):
    month = date.month
    day = date.day
    df = dataset[(dataset['month']==month) & (dataset['day']<=day) & (dataset['day']>=day-3)]
    # Independent variable data selection
    X = df.iloc[:,[2,6,8,19]].values
    # Dependent variable data selection
    y = df.iloc[:,11].values
    # spliting data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)
    # Fitting Random Forest Regression to the dataset
    for_reg = RandomForestRegressor(n_estimators = 10, random_state = 0)
    for_reg.fit(X_train, y_train)
    # Calculating the mean values of Independent variables Data
    dew=np.mean(X[:,0])                         # Dew point
    hum=np.mean(X[:,1])                         # Humidty
    press=np.mean(X[:,2])                       # Pressure
    windsp=np.mean(X[:,3])                      # Wind speed
    # Creating a 2D numpy array of shape(1, 4)
    test = np.array([dew,hum,press,windsp]).reshape((1,4))
    # Predicting a new result
    temperature = for_reg.predict(test)
    # returning the predicted temperature
    return int(round(*temperature))

@server.route('/')
def mydashapp():
    return myapp.index()

if __name__ == "__main__":
    server.run()
