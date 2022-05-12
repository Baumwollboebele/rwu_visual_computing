    gblur = cv.GaussianBlur(gray, (5, 5), sigmaX=0, sigmaY=0)
    edged = cv.Canny(gblur, 75, 200)

    contours = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv.contourArea, reverse = True)

    screenCnt = np.zeros(1)

    for c in contours:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    if len(screenCnt) == 1:
        print("Nothing found")
        exit()

    src_pts = np.array(screenCnt, dtype=np.float32).reshape(4,2)

    dst_pts = np.array([[0, 0],   [w-1, 0],  [w-1, h-1], [0, h-1]], dtype=np.float32)

    src_pts = order_points(src_pts)

    M = cv.getPerspectiveTransform(src_pts, dst_pts)
    warp = cv.warpPerspective(gray, M, (w, h))

    adaptive_binary_image = cv.adaptiveThreshold(warp,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,151,10)
    tresh, binary_image = cv.threshold(warp, 125,255,cv.THRESH_BINARY)


def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")

    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)

    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect